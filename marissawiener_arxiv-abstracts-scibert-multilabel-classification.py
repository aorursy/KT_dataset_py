# utils

import numpy as np

import pandas as pd

import json

import string

import matplotlib.pyplot as plt

from datetime import datetime

# modeling utils

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score, 

    f1_score, 

    classification_report

)

# BERT

from transformers import BertTokenizer, BertForSequenceClassification, AdamW

import torch

import torch.nn as nn

from torch.utils.data import DataLoader
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def pull_data_and_dedupe(n_obs = -1):

    data  = []

    with open("../input/arxiv/arxiv-metadata-oai-snapshot.json", 'r') as f:

        for line in f: 

            data.append(json.loads(line))

    data = pd.DataFrame(data)

    

    if n_obs > 0:

        data = data.head((n_obs + 1000))# to account for deduping and withdrawn papers

    

    keep_cols = ['id', 'title', 'abstract', 'categories']

    data = data[keep_cols]

    # remove abstracts from withdrawn records

    data = data[data['abstract'].str.contains('paper has been withdrawn') == False]

    

    # create columns for each unique title

    categories = set([i for l in [x.split(' ') for x in data['categories']] for i in l])

    for un in categories:

        data[un] = np.where(data['categories'].str.contains(un), 1, 0)

    # remove duplicate records which contain different flags

    data = data.drop(columns = 'categories').groupby(by = ['id', 'title', 'abstract'],as_index = False).max()

    

    data.reset_index(drop = True, inplace = True)

    if n_obs > 0:

        data = data.head(n_obs)

    

    return data, categories
def abstract_prep(data):

    # lower abstract and remove numbers, punctuation, and special characters

    #metadata['abstract'] = [a.strip() for a in metadata['abstract']]

    data['abstract'] = [a.lower().strip() for a in data['abstract']]

    data['abstract'] = data['abstract'].str.replace('\n', ' ', regex = False).str.replace(r'\s\s+', ' ', regex = True)

    data['abstract'] = data['abstract'].str.replace('([.,!?()])', r' \1 ')

    return data
# generate counts by label

def handle_category_counts(data, show_plot = True):

    category_count = data[data.columns[3:].to_list()].sum()

    

    if show_plot:

        # create plot

        labs = category_count.sort_values(ascending = False).index.to_list()

        counts = category_count.sort_values(ascending = False).values

        fig = plt.figure(figsize=(16, 9))

        ax = fig.gca()

        fig.suptitle('Bottom 50 Categories', fontsize=20)

        plt.ylabel('Number of Papers', fontsize=14)

        plt.xlabel('Labeled Paper Category', fontsize=14)

        plt.setp(ax.get_xticklabels(), ha="right", rotation=45, fontsize=14) # Specify a rotation for the tick labels 

        plt.bar(labs[(len(category_count)-50):],counts[(len(category_count)-50):])

        plt.show()



        # plot sentence lengths to determine max length

        abstract_lengths = [len(t.split()) for t in data['abstract']]

        abstract_lengths.sort()

        plt.hist([i for i in abstract_lengths if i > 299 and i < 400])

        plt.show()

    

    # determine which categories to preserve and update dataframe

    categories = list(category_count[category_count > 15].index)

    # remove columns pertaining to categories to exclude  

    data = data[list(data.columns[:3]) + categories]

    print('\tfinal df size:', data.shape)

    

    # isolate data

    labels = data.loc[:, categories].values

    input_data = data[['id', 'abstract']]

    

    return metadata, labels, input_data, categories

# fixed parameters and hyperparameters for dataset creation and model training

class Config:

    def __init__(self, categories):

        # allow class to be inherited

        super(Config, self).__init__()

        

        # general parameters

        self.SEED = 9

        self.MODEL_PATH = "allenai/scibert_scivocab_uncased"

        self.NUM_LABELS = len(categories)

        

        # load tokenizer and set related parameters

        self.TOKENIZER = BertTokenizer.from_pretrained(self.MODEL_PATH)

        self.MAX_LENGTH = 350 # from EDA

        

        # determine optimal batch size based on 

        self.N_GPU = torch.cuda.device_count()

        if self.N_GPU == 0:

            self.N_GPU = 1

        self.BATCH_SIZE = self.N_GPU * 8

            

        # validation & test split

        self.VALIDATION_SPLIT = .3

        

        # set model parameters

        self.LR = 3e-5

        self.CRITERION = nn.BCEWithLogitsLoss()

        self.EPOCHS = 3

        

        # model selection

        self.SELECTION_METRIC = 'loss' # options: f1 accuracy loss
# dataset object

class arxiv_dataset(torch.utils.data.Dataset):

            

    def __init__(self, abstrcts, lbls, msks):

        self.abstrcts = torch.Tensor(abstrcts).long()

        self.msks = torch.Tensor(msks).long()

        # cast labels as float

        self.lbls = torch.Tensor(lbls).float()

        

    def __len__(self):

        return self.lbls.shape[0]

    

    def __getitem__(self, index):

        abstracts_ = self.abstrcts[index, :]

        labels_ = self.lbls[index, :]

        masks_ = self.msks[index, :]

        return abstracts_, labels_, masks_
# tokenize, split data, create dataloader objects from arxiv dataset objects

def bert_prep(input_data, labels, config):

    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)

    tokenized_abstracts = tokenizer.batch_encode_plus(

                input_data['abstract'],

                max_length = config.MAX_LENGTH,

                pad_to_max_length = True,

                truncation = True,

                return_attention_mask = True,

                return_token_type_ids = False,

                return_tensors = 'pt'

            )



    # initial train and test split

    token_train, token_test, mask_train, mask_test, \

    y_train, y_test = train_test_split(np.array(tokenized_abstracts['input_ids']),

                                       np.array(tokenized_abstracts['attention_mask']), 

                                       np.array(labels), 

                                       test_size = config.VALIDATION_SPLIT,

                                       random_state = config.SEED)

    # split test into test and validation

    token_val, token_test, mask_val, mask_test, \

    y_val, y_test = train_test_split(token_test,

                                     mask_test,

                                     y_test,

                                     test_size = 0.5,

                                     random_state = config.SEED)



    # qc check

    print('QC......')

    print('\tabstract - training:', token_train.shape)

    print('\tmask - training:', mask_train.shape)

    print('\tlabel - training:', y_train.shape)

    print('\tabstract - validation:', token_val.shape)

    print('\tmask - valdiation:', mask_val.shape)

    print('\tlabel - validation:', y_val.shape)

    print('\tabstract - test:', token_test.shape)

    print('\tmask - test:', mask_test.shape)

    print('\tlabel - test:', y_test.shape)





    ## create datasets and loaders

    train_data = arxiv_dataset(token_train, y_train, mask_train)

    val_data = arxiv_dataset(token_val, y_val, mask_val)

    test_data = arxiv_dataset(token_test, y_test, mask_test)



    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE)

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCH_SIZE)

    

    return train_dataloader, val_dataloader, test_dataloader
# train and validate model 

def train_model(train_dataloader, val_dataloader, test_dataloader, config):

    # intitialize model

    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=config.NUM_LABELS)

    if torch.cuda.is_available():

        model = model.cuda() 

        

    # set optimizer

    param_optimizer = list(model.named_parameters())

    # According to the huggingface recommendations

    # weight decay is set to 0 for bias layers

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

                                     'weight_decay_rate': 0.01},

                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

                                     'weight_decay_rate': 0.0}]

    # Using BERT's Adam optimizer similar to the original Tensorflow optimizer

    optimizer = AdamW(optimizer_grouped_parameters,

                      lr = config.LR,

                      weight_decay = 0.01,

                      correct_bias = False)

    

    ## initialize values

    if config.N_GPU > 1:

        model = nn.DataParallel(model)

    

    epoch_train_loss = []

    epoch_valid_loss = []

    epoch_valid_f1 = []

    epoch_valid_accuracy = []

    epoch_test_loss = []

    epoch_test_f1 = []

    epoch_test_accuracy = []



    ## Training/Validation Loop

    for epoch in range(1, config.EPOCHS + 1):

        print('\tEPOCH:', epoch)



        train_loss = 0.0

        valid_loss = 0.0

        test_loss = 0.0



        ######### TRAINING #############

        # set model to train mode

        model.train()



        #batch = 1

        # iterate through each observation

        for data in train_dataloader:

            #print('BATCH:', batch)

            abstracts_, labels_, masks_ = data



            # move data to GPU

            if torch.cuda.is_available():

                abstracts_ = abstracts_.cuda()

                masks_ = masks_.cuda()

                labels_ = labels_.cuda()



            # zero out optimizer gradients

            optimizer.zero_grad()



            # fit model and calculate loss

            logits = model(input_ids = abstracts_, attention_mask = masks_)[0]

            loss = config.CRITERION(logits, labels_)



            if config.N_GPU > 1 :

                loss = loss.mean()



            loss.backward()

            optimizer.step()



            train_loss += loss.item()

            #batch += 1



        epoch_t_loss = train_loss/len(train_dataloader)

        print(f"\t\tTrain loss: {epoch_t_loss}")



        ###### VALIDATION ########

        # set model to train mode

        model.eval()



        valid_truth = []

        valid_preds = []



        # iterate through each observation

        for data in val_dataloader:

            #print('BATCH:', batch)

            abstracts_, labels_, masks_ = data

            # move data to GPU

            if torch.cuda.is_available():

                abstracts_ = abstracts_.cuda()

                masks_ = masks_.cuda()

                labels_ = labels_.cuda()



            # fit model and calculate loss

            logits = model(input_ids = abstracts_, attention_mask = masks_)[0]

            loss = config.CRITERION(logits, labels_)



            if config.N_GPU > 1 :

                loss = loss.mean()



            valid_loss += loss.item()

            #batch += 1



            # since we're using BCEWithLogitsLoss, to get the predictions -

            # - sigmoid has to be applied on the logits first

            logits_cpu = torch.sigmoid(logits)

            logits_cpu = np.round(logits_cpu.cpu().detach().numpy())

            labels_cpu = labels_.cpu().numpy()



            # keep list of outputs for validation

            valid_truth.extend(labels_cpu)

            valid_preds.extend(logits_cpu)

            

        ### Validation Metrics 

        epoch_v_loss = valid_loss/len(val_dataloader)

        print(f"\t\tValid loss: {epoch_v_loss}")



        epoch_v_accuracy_score = accuracy_score(valid_truth,valid_preds)

        print('\t\tVal Accuracy:', epoch_v_accuracy_score)



        epoch_v_micro_f1_score = f1_score(valid_truth,valid_preds, average='micro')

        print('\t\tVal Micro F1 score:', epoch_v_micro_f1_score)

            

        ###### TEST ########

        test_truth = []

        test_preds = []



        # iterate through each observation

        for data in test_dataloader:

            #print('BATCH:', batch)

            abstracts_, labels_, masks_ = data

            # move data to GPU

            if torch.cuda.is_available():

                abstracts_ = abstracts_.cuda()

                masks_ = masks_.cuda()

                labels_ = labels_.cuda()



            # fit model and calculate loss

            logits = model(input_ids = abstracts_, attention_mask = masks_)[0]

            loss = config.CRITERION(logits, labels_)



            if config.N_GPU > 1 :

                loss = loss.mean()



            test_loss += loss.item()

            #batch += 1



            # since we're using BCEWithLogitsLoss, to get the predictions -

            # - sigmoid has to be applied on the logits first

            logits_cpu = torch.sigmoid(logits)

            logits_cpu = np.round(logits_cpu.cpu().detach().numpy())

            labels_cpu = labels_.cpu().numpy()



            # keep list of outputs for validation

            test_truth.extend(labels_cpu)

            test_preds.extend(logits_cpu)



        

        epoch_tst_loss = test_loss/len(test_dataloader)

        print(f"\t\tTest loss: {epoch_tst_loss}")



        epoch_tst_accuracy_score = accuracy_score(test_truth,test_preds)

        print('\t\tTest Accuracy:', epoch_tst_accuracy_score)



        epoch_tst_micro_f1_score = f1_score(test_truth,test_preds, average='micro')

        print('\t\tTest Micro F1 score:', epoch_tst_micro_f1_score)



        # if validation selection metric improved, set the best model to the current model

        if config.SELECTION_METRIC == 'loss':

            if len(epoch_valid_loss) == 0:

                best_model = model

                best_epoch = epoch

            else:

                if epoch_valid_loss[-1] > epoch_v_loss:

                    print('\t\tReplace model with version from epoch', epoch, 'based on', config.SELECTION_METRIC)

                    best_model = model

                    best_epoch = epoch



        if config.SELECTION_METRIC == 'f1':

            if len(epoch_valid_f1) == 0:

                best_model = model

                best_epoch = epoch

            else:

                if epoch_valid_f1[-1] > epoch_v_micro_f1_score:

                    print('\t\tReplace model with version from epoch', epoch, 'based on', config.SELECTION_METRIC)

                    best_model = model

                    best_epoch = epoch



        if config.SELECTION_METRIC == 'accuracy':

            if len(epoch_valid_accuracy) == 0:

                best_model = model

                best_epoch = epoch

            else:

                if epoch_valid_accuracy[-1] > epoch_v_accuracy_score:

                    print('\t\tReplace model with version from epoch', epoch, 'based on', config.SELECTION_METRIC)

                    best_model = model

                    best_epoch = epoch



        # update epoch loss lists

        epoch_train_loss.append(epoch_t_loss)

        epoch_valid_loss.append(epoch_v_loss)

        epoch_valid_f1.append(epoch_v_micro_f1_score)

        epoch_valid_accuracy.append(epoch_v_accuracy_score)

        epoch_test_loss.append(epoch_tst_loss)

        epoch_test_f1.append(epoch_tst_micro_f1_score)

        epoch_test_accuracy.append(epoch_tst_accuracy_score)



    tracker_df = pd.DataFrame({'epoch' : list(range(1,config.EPOCHS + 1)),

                               'train_loss' : epoch_train_loss,

                               'validation_loss' : epoch_valid_loss,

                               'validation_accuracy' : epoch_valid_accuracy,

                               'validation_f1' : epoch_valid_f1,

                               'test_loss' : epoch_test_loss,

                               'test_accuracy' : epoch_test_accuracy,

                               'test_f1' : epoch_test_f1})

    tracker_df['best_model_indicator'] = np.where(tracker_df['epoch'] == best_epoch, 1, 0)

    

    return best_model, tracker_df, valid_truth, valid_preds, test_truth, test_preds
def run(metadata, categories, n_obs):

    print('*********************************************** START: Fitting Model with', n_obs, 'Obs ***********************************************')

    print('Prepping data.....')

    metadata = abstract_prep(metadata)

    

    print('EDA.....')

    metadata, labels, input_data, categories = handle_category_counts(metadata, show_plot = False)

    

    print('Prepping data for BERT.....')

    config = Config(categories)

    train_dataloader, val_dataloader, test_dataloader = bert_prep(input_data, labels, config)

    

    print('Training and validating BERT.....')

    best_model, tracker_df, valid_truth, \

    valid_preds, test_truth, test_preds = train_model(train_dataloader, val_dataloader, test_dataloader, config)

    

    print('Saving best performing model.....')

    torch.save(best_model.state_dict(), datetime.now().strftime('%Y%m%d') + '_' + str(n_obs) + '_' + str(config.EPOCHS) + '.pt')

    print('\tsaved model: ' + datetime.now().strftime('%Y%m%d') + '_' + str(n_obs) + '.pt')

    

    print('*********************************************** END: Fitting Model with', n_obs, 'Obs ***********************************************')

    return best_model, train_dataloader, val_dataloader, test_dataloader, tracker_df, valid_truth, valid_preds, test_truth, test_preds, config, tracker_df
############ SET PARAMS ##############

n_obs = 200000
%%time

metadata, categories = pull_data_and_dedupe(n_obs)
%%time

best_model, train_dataloader, val_dataloader, test_dataloader, tracker_df, valid_truth, valid_preds, test_truth, test_preds, config, tracker_df = run(metadata, categories, n_obs) # test
tracker_df.to_csv('tracker_df_' + datetime.now().strftime('%Y%m%d') + '_' + str(n_obs) + '_' + str(config.EPOCHS) + '.csv')