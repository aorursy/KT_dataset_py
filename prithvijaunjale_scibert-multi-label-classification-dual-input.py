# ! pip3 install transformers
import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

import re

import copy

from tqdm.notebook import tqdm

import gc



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data import Dataset, DataLoader



from sklearn.metrics import (

    accuracy_score, 

    f1_score, 

    classification_report

)



from transformers import (

    AutoTokenizer, 

    AutoModel,

    get_linear_schedule_with_warmup

)



project_dir = '../input/avjanatahackresearcharticlesmlc/av_janatahack_data/'
! nvidia-smi
train_df = pd.read_csv(project_dir + 'train.csv')

train_df.head()
# preprocessing

def clean_text(text):

    text = text.split()

    text = [x.strip() for x in text]

    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]

    text = ' '.join(text)

    text = re.sub('([.,!?()])', r' \1 ', text)

    return text

    



def get_texts(df):

    titles = df['TITLE'].apply(clean_text)

    titles = titles.values.tolist()

    abstracts = df['ABSTRACT'].apply(clean_text)

    abstracts = abstracts.values.tolist()

    return titles, abstracts





def get_labels(df):

    labels = df.iloc[:, 3:].values

    return labels



titles, abstracts = get_texts(train_df)

labels = get_labels(train_df)



for t, a, l in zip(titles[:5], abstracts[:5], labels[:5]):

    print(f'TITLE -\t{t}')

    print(f'ABSTRACT -\t{a}')

    print(f'LABEL -\t{l}')

    print('_' * 80)

    print()
# no. of samples for each class

categories = train_df.columns.to_list()[3:]

plt.figure(figsize=(6, 4))



ax = sns.barplot(categories, train_df.iloc[:, 3:].sum().values)

plt.ylabel('Number of papers')

plt.xlabel('Paper type ')

plt.xticks(rotation=90)

plt.show()
# no of samples having multiple labels

row_sums = train_df.iloc[:, 3:].sum(axis=1)

multilabel_counts = row_sums.value_counts()



plt.figure(figsize=(6, 4))

ax = sns.barplot(multilabel_counts.index, multilabel_counts.values)

plt.ylabel('Number of papers')

plt.xlabel('Number of labels')

plt.show()
# title lengths

y = [len(t.split()) for t in titles]

x = range(0, len(y))

plt.bar(x, y)
# abstracts lengths

y = [len(t.split()) for t in abstracts]

x = range(0, len(y))

plt.bar(x, y)
class Config:

    def __init__(self):

        super(Config, self).__init__()



        self.SEED = 42

        self.MODEL_PATH = 'allenai/scibert_scivocab_uncased'

        self.NUM_LABELS = 6



        # data

        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)

        self.MAX_LENGTH1 = 20

        self.MAX_LENGTH2 = 320

        self.BATCH_SIZE = 5

        self.VALIDATION_SPLIT = 0.25



        # model

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.FULL_FINETUNING = True

        self.LR = 3e-5

        self.OPTIMIZER = 'AdamW'

        self.CRITERION = 'BCEWithLogitsLoss'

        self.SAVE_BEST_ONLY = True

        self.N_VALIDATE_DUR_TRAIN = 3

        self.EPOCHS = 1



config = Config()
class TransformerDataset(Dataset):

    def __init__(self, df, indices, set_type=None):

        super(TransformerDataset, self).__init__()



        df = df.iloc[indices]

        self.titles, self.abstracts = get_texts(df)

        self.set_type = set_type

        if self.set_type != 'test':

            self.labels = get_labels(df)



        self.max_length1 = config.MAX_LENGTH1

        self.max_length2 = config.MAX_LENGTH2

        self.tokenizer = config.TOKENIZER



    def __len__(self):

        return len(self.titles)

    

    def __getitem__(self, index):

        tokenized_titles = self.tokenizer.encode_plus(

            self.titles[index], 

            max_length=self.max_length1,

            pad_to_max_length=True,

            truncation=True,

            return_attention_mask=True,

            return_token_type_ids=False,

            return_tensors='pt'

        )

        input_ids_titles = tokenized_titles['input_ids'].squeeze()

        attention_mask_titles = tokenized_titles['attention_mask'].squeeze()

        

        tokenized_abstracts = self.tokenizer.encode_plus(

            self.abstracts[index], 

            max_length=self.max_length2,

            pad_to_max_length=True,

            truncation=True,

            return_attention_mask=True,

            return_token_type_ids=False,

            return_tensors='pt'

        )

        input_ids_abstracts = tokenized_abstracts['input_ids'].squeeze()

        attention_mask_abstracts = tokenized_abstracts['attention_mask'].squeeze()



        if self.set_type != 'test':

            return {

                'titles': {

                    'input_ids': input_ids_titles.long(),

                    'attention_mask': attention_mask_titles.long(),

                },

                'abstracts': {

                    'input_ids': input_ids_abstracts.long(),

                    'attention_mask': attention_mask_abstracts.long(),

                },

                'labels': torch.Tensor(self.labels[index]).float(),

            }



        return {

            'titles': {

                'input_ids': input_ids_titles.long(),

                'attention_mask': attention_mask_titles.long(),

            },

            'abstracts': {

                'input_ids': input_ids_abstracts.long(),

                'attention_mask': attention_mask_abstracts.long(),

            }

        }
# train-val split



np.random.seed(config.SEED)



dataset_size = len(train_df)

indices = list(range(dataset_size))

split = int(np.floor(config.VALIDATION_SPLIT * dataset_size))

np.random.shuffle(indices)



train_indices, val_indices = indices[split:], indices[:split]
train_data = TransformerDataset(train_df, train_indices)

val_data = TransformerDataset(train_df, val_indices)



train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)

val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)



b = next(iter(train_dataloader))

for k, v in b.items():

    if k == 'titles' or k == 'abstracts':

        print(k)

        for k_, v_ in b[k].items():

            print(f'{k_} shape: {v_.shape}\n')

    else:

        print(f'{k} shape: {v.shape}')
class DualSciBert(nn.Module):

    def __init__(self):

        super(DualSciBert, self).__init__()



        self.titles_model = AutoModel.from_pretrained(config.MODEL_PATH)

        self.abstracts_model = AutoModel.from_pretrained(config.MODEL_PATH)

        

        self.dropout = nn.Dropout(0.25)

        self.avgpool = nn.AvgPool1d(2, 2)

        self.output = nn.Linear(768, config.NUM_LABELS)



    def forward(

        self,

        input_ids_titles, 

        attention_mask_titles=None, 

        input_ids_abstracts=None,

        attention_mask_abstracts=None

        ):

        

        _, titles_features = self.titles_model(

            input_ids=input_ids_titles,

            attention_mask=attention_mask_titles

        )

        titles_features = titles_features.unsqueeze(1)

        titles_features_pooled = self.avgpool(titles_features)

        titles_features_pooled = titles_features_pooled.squeeze(1)

        

        _, abstracts_features = self.abstracts_model(

            input_ids=input_ids_abstracts,

            attention_mask=attention_mask_abstracts

        )

        abstracts_features = abstracts_features.unsqueeze(1)

        abstracts_features_pooled = self.avgpool(abstracts_features)

        abstracts_features_pooled = abstracts_features_pooled.squeeze(1)

        

        combined_features = torch.cat((

            titles_features_pooled, 

            abstracts_features_pooled), 

            dim=1

        )

        x = self.dropout(combined_features)

        x = self.output(x)

        

        return x
class SiameseSciBert(nn.Module):

    def __init__(self):

        super(SiameseSciBert, self).__init__()



        self.model = AutoModel.from_pretrained(config.MODEL_PATH)

        self.dropout = nn.Dropout(0.25)

        self.avgpool = nn.AvgPool1d(2, 2)

        self.output = nn.Linear(768, config.NUM_LABELS)



    def forward(

        self,

        input_ids_titles, 

        attention_mask_titles=None, 

        input_ids_abstracts=None,

        attention_mask_abstracts=None

        ):

        

        _, titles_features = self.model(

            input_ids=input_ids_titles,

            attention_mask=attention_mask_titles

        )

        titles_features = titles_features.unsqueeze(1)

        titles_features_pooled = self.avgpool(titles_features)

        titles_features_pooled = titles_features_pooled.squeeze(1)

        

        _, abstracts_features = self.model(

            input_ids=input_ids_abstracts,

            attention_mask=attention_mask_abstracts

        )

        abstracts_features = abstracts_features.unsqueeze(1)

        abstracts_features_pooled = self.avgpool(abstracts_features)

        abstracts_features_pooled = abstracts_features_pooled.squeeze(1)

        

        combined_features = torch.cat((

            titles_features_pooled, 

            abstracts_features_pooled), 

            dim=1

        )

        x = self.dropout(combined_features)

        x = self.output(x)

        

        return x
class SiameseSciBertRNN(nn.Module):

    def __init__(self):

        super(SiameseSciBertRNN, self).__init__()



        self.model = AutoModel.from_pretrained(config.MODEL_PATH)

        self.dropout = nn.Dropout(0.3)

        self.avgpool = nn.AvgPool1d(2, 2)

        self.maxpool = nn.MaxPool1d(2, 2)

        

        self.rnn = nn.GRU(

            input_size=768, 

            hidden_size=128, 

            batch_first=True,

            bidirectional=True,

        )

        

        self.output = nn.Linear(256, config.NUM_LABELS)



    def forward(

        self,

        input_ids_titles, 

        attention_mask_titles=None, 

        input_ids_abstracts=None,

        attention_mask_abstracts=None

        ):

        

        titles_hidden_states, _ = self.model(

            input_ids=input_ids_titles,

            attention_mask=attention_mask_titles

        )

        self.rnn.flatten_parameters()

        titles_rnn_out, _ = self.rnn(titles_hidden_states)

        titles_rnn_feat = titles_rnn_out.mean(dim=1)

        titles_rnn_feat = titles_rnn_feat.unsqueeze(1)

        titles_rnn_feat_pooled = self.avgpool(titles_rnn_feat)

        titles_rnn_feat_pooled = titles_rnn_feat_pooled.squeeze(1)

        

        abstracts_hidden_states, _ = self.model(

            input_ids=input_ids_abstracts,

            attention_mask=attention_mask_abstracts

        )

        self.rnn.flatten_parameters()

        abstracts_rnn_out, _ = self.rnn(abstracts_hidden_states)

        abstracts_rnn_feat = abstracts_rnn_out.mean(dim=1)

        abstracts_rnn_feat = abstracts_rnn_feat.unsqueeze(1)

        abstracts_rnn_feat_pooled = self.avgpool(abstracts_rnn_feat)

        abstracts_rnn_feat_pooled = abstracts_rnn_feat_pooled.squeeze(1)



        combined_features = torch.cat((

            titles_rnn_feat_pooled, 

            abstracts_rnn_feat_pooled), 

            dim=1

        )

        x = self.dropout(combined_features)

        x = self.output(x)

        

        return x
device = config.DEVICE

device
def val(model, val_dataloader, criterion):

    

    val_loss = 0

    true, pred = [], []

    

    # set model.eval() every time during evaluation

    model.eval()

    

    for step, batch in enumerate(val_dataloader):

        # unpack the batch contents and push them to the device (cuda or cpu).

        b_input_ids_titles = batch['titles']['input_ids'].to(device)

        b_attention_mask_titles = batch['titles']['attention_mask'].to(device)

        b_input_ids_abstracts = batch['abstracts']['input_ids'].to(device)

        b_attention_mask_abstracts = batch['abstracts']['attention_mask'].to(device)

        b_labels = batch['labels'].to(device)



        # using torch.no_grad() during validation/inference is faster -

        # - since it does not update gradients.

        with torch.no_grad():

            # forward pass

            logits = model(

                b_input_ids_titles, 

                b_attention_mask_titles,

                b_input_ids_abstracts,

                b_attention_mask_abstracts

            )

            

            # calculate loss

            loss = criterion(logits, b_labels)

            val_loss += loss.item()



            # since we're using BCEWithLogitsLoss, to get the predictions -

            # - sigmoid has to be applied on the logits first

            logits = torch.sigmoid(logits)

            logits = np.round(logits.cpu().numpy())

            labels = b_labels.cpu().numpy()

            

            # the tensors are detached from the gpu and put back on -

            # - the cpu, and then converted to numpy in order to -

            # - use sklearn's metrics.



            pred.extend(logits)

            true.extend(labels)



    avg_val_loss = val_loss / len(val_dataloader)

    print('Val loss:', avg_val_loss)

    print('Val accuracy:', accuracy_score(true, pred))



    val_micro_f1_score = f1_score(true, pred, average='micro')

    print('Val micro f1 score:', val_micro_f1_score)

    return val_micro_f1_score





def train(

    model, 

    train_dataloader, 

    val_dataloader, 

    criterion, 

    optimizer, 

    scheduler, 

    epoch

    ):

    

    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop

    nv = config.N_VALIDATE_DUR_TRAIN

    temp = len(train_dataloader) // nv

    temp = temp - (temp % 100)

    validate_at_steps = [temp * x for x in range(1, nv + 1)]

    

    train_loss = 0

    for step, batch in enumerate(tqdm(train_dataloader, 

                                      desc='Epoch ' + str(epoch))):

        # set model.eval() every time during training

        model.train()

        

        # unpack the batch contents and push them to the device (cuda or cpu).

        b_input_ids_titles = batch['titles']['input_ids'].to(device)

        b_attention_mask_titles = batch['titles']['attention_mask'].to(device)

        b_input_ids_abstracts = batch['abstracts']['input_ids'].to(device)

        b_attention_mask_abstracts = batch['abstracts']['attention_mask'].to(device)

        b_labels = batch['labels'].to(device)



        # clear accumulated gradients

        optimizer.zero_grad()



        # forward pass

        logits = model(

            b_input_ids_titles, 

            b_attention_mask_titles,

            b_input_ids_abstracts,

            b_attention_mask_abstracts

        )

        

        # calculate loss

        loss = criterion(logits, b_labels)

        train_loss += loss.item()



        # backward pass

        loss.backward()



        # update weights

        optimizer.step()

        

        # update scheduler

        scheduler.step()



        if step in validate_at_steps:

            print(f'-- Step: {step}')

            _ = val(model, val_dataloader, criterion)

    

    avg_train_loss = train_loss / len(train_dataloader)

    print('Training loss:', avg_train_loss)
def run():

    # setting a seed ensures reproducible results.

    # seed may affect the performance too.

    torch.manual_seed(config.SEED)



    criterion = nn.BCEWithLogitsLoss()

    

    # define the parameters to be optmized -

    # - and add regularization

    if config.FULL_FINETUNING:

        param_optimizer = list(model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_parameters = [

            {

                "params": [

                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)

                ],

                "weight_decay": 0.001,

            },

            {

                "params": [

                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)

                ],

                "weight_decay": 0.0,

            },

        ]

        optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)



    num_training_steps = len(train_dataloader) * config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_training_steps

    )



    max_val_micro_f1_score = float('-inf')

    for epoch in range(config.EPOCHS):

        train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch)

        val_micro_f1_score = val(model, val_dataloader, criterion)



        if config.SAVE_BEST_ONLY:

            if val_micro_f1_score > max_val_micro_f1_score:

                best_model = copy.deepcopy(model)

                best_val_micro_f1_score = val_micro_f1_score



                model_name = 'scibertfft_dualinput_best_model'

                torch.save(best_model.state_dict(), model_name + '.pt')



                print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')

                max_val_micro_f1_score = val_micro_f1_score



    return best_model, best_val_micro_f1_score
model = DualSciBert()

model.to(device);

best_model, best_val_micro_f1_score = run()
del model

model = SiameseSciBert()

model.to(device);

best_model, best_val_micro_f1_score = run()
del model

model = SiameseSciBertRNN()

model.to(device);

best_model, best_val_micro_f1_score = run()
test_df = pd.read_csv(project_dir + 'test.csv')

dataset_size = len(test_df)

test_indices = list(range(dataset_size))



test_data = TransformerDataset(test_df, test_indices, set_type='test')

test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE)



def predict(model):

    val_loss = 0

    test_pred = []

    model.eval()

    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        b_input_ids_titles = batch['titles']['input_ids'].to(device)

        b_attention_mask_titles = batch['titles']['attention_mask'].to(device)

        b_input_ids_abstracts = batch['abstracts']['input_ids'].to(device)

        b_attention_mask_abstracts = batch['abstracts']['attention_mask'].to(device)



        with torch.no_grad():

            logits = model(

                b_input_ids_titles, 

                b_attention_mask_titles,

                b_input_ids_abstracts,

                b_attention_mask_abstracts

            )

            logits = torch.sigmoid(logits)

            logits = np.round(logits.cpu().numpy())

            test_pred.extend(logits)



    test_pred = np.array(test_pred)

    return test_pred



# test_pred = predict(best_model)
def submit():

    sample_submission = pd.read_csv(project_dir + 'sample_submission.csv')

    ids = sample_submission['ID'].values.reshape(-1, 1)



    merged = np.concatenate((ids, test_pred), axis=1)

    submission = pd.DataFrame(merged, columns=sample_submission.columns).astype(int)

    return submission



# submission = submit()
# submission_fname = f'submission_scibertfft_dualinput_microf1-{round(best_val_micro_f1_score, 4)}.csv'

# submission.to_csv(submission_fname, index=False)