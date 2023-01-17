import pandas as pd
import numpy as np
import random

random_seed = 0
random.seed(random_seed)
dataset_filepath = '../input/60k-stack-overflow-questions-with-quality-rate/data.csv'
rawDf = pd.read_csv(dataset_filepath)
rawDf.head()
prunedDf = rawDf.drop(columns=['Id', 'Tags', 'CreationDate'])
prunedDf.head()
import en_core_web_sm
import re
import string
from bs4 import BeautifulSoup

nlp = en_core_web_sm.load()

def Tokenize(text, nlp):
    text = text.lower()
    text = re.sub("\d+", "", text) # Remove all digits
    text = BeautifulSoup(text).get_text() # Remove markups    
    tokens = [token.text for token in nlp.tokenizer(text) if not token.text.isspace()]
    return tokens
title_maximum_length = 0
body_maximum_length = 0

for index, row in prunedDf.iterrows():
    title = row['Title']
    body = row['Body']
    tokenized_title = Tokenize(title, nlp)
    tokenized_body = Tokenize(body, nlp)
    """if index % 25000 == 0:
        print ("body = {}".format(body))
        print ("tokenized_body = {}".format(tokenized_body))
    """
    prunedDf.loc[index, 'Title'] = tokenized_title
    prunedDf.loc[index, 'Body'] = tokenized_body
    if len(tokenized_title) > title_maximum_length:
        title_maximum_length = len(tokenized_title)
    if len(tokenized_body) > body_maximum_length:
        body_maximum_length = len(tokenized_body)
featuresDf = prunedDf.drop(columns=['Y'])
targetDf = prunedDf['Y']

from sklearn.model_selection import train_test_split
train_features_Df, validation_features_Df, train_target_Df, validation_target_Df = train_test_split(featuresDf, targetDf, test_size=0.2, random_state=random_seed)
word_to_occurrences_dict = {}
for index, row in train_features_Df.iterrows():
    title_words = row['Title']
    body_words = row['Body']
    for word in title_words:
        if word in word_to_occurrences_dict:
            word_to_occurrences_dict[word] += 1
        else:
            word_to_occurrences_dict[word] = 1
    for word in body_words:
        if word in word_to_occurrences_dict:
            word_to_occurrences_dict[word] += 1
        else:
            word_to_occurrences_dict[word] = 1
    if index % 1000 == 0:
        print(".", end="", flush=True)
print("Before filtering the single occurrences, len(word_to_occurrences_dict) = {}".format(len(word_to_occurrences_dict)))
single_occurrence_words = []
for word, occurrences in word_to_occurrences_dict.items():
    if occurrences < 20:
        single_occurrence_words.append(word)
for word in single_occurrence_words:
    word_to_occurrences_dict.pop(word)
print("After filtering the single occurrences, len(word_to_occurrences_dict) = {}".format(len(word_to_occurrences_dict)))
sorted_tokens = sorted(word_to_occurrences_dict.items(),
                           key=lambda x: x[1], reverse=True) # Cf. https://careerkarma.com/blog/python-sort-a-dictionary-by-value/
sorted_tokens = [('ENDOFSEQ', 0), ('UNKNOWN', 0), ('NOTSET', 0)] + sorted_tokens
print(sorted_tokens[0:100])
def WordToIndex(token_occurrence_pairs):
    word_to_index_dict = {}
    index_to_word_dict = {}
    for index, token_occurrence in enumerate(token_occurrence_pairs):
        word_to_index_dict[token_occurrence[0]] = index
        index_to_word_dict[index] = token_occurrence[0]
    return word_to_index_dict, index_to_word_dict

word_to_index_dict, index_to_word_dict = WordToIndex(sorted_tokens)
print ("word_to_index_dict['compile'] = {}".format(word_to_index_dict['compile']))
print ("index_to_word_dict[391] = {}".format(index_to_word_dict[391]))
title_maximum_length += 1 # To make room for the extra 'ENDOFSEQ' token
body_maximum_length += 1

def ConvertTokensListToIndices(tokens, word_to_index_dict, maximum_length):
    indices = [word_to_index_dict['NOTSET']] * maximum_length
    for tokenNdx, token in enumerate(tokens):
        index = word_to_index_dict.get(token, word_to_index_dict['UNKNOWN']) # If the word is not in the dictionary, fall back to 'UNKOWN'
        indices[tokenNdx] = index
    if len(tokens) < maximum_length:
        indices[len(tokens)] = word_to_index_dict['ENDOFSEQ']
    return indices
train_feature_indices_Df = pd.DataFrame(columns=['Title', 'Body'])
validation_feature_indices_Df = pd.DataFrame(columns=['Title', 'Body'])
for row in train_features_Df.itertuples():
    titleList = row[1]
    bodyList = row[2]
    title_indices = ConvertTokensListToIndices(titleList, word_to_index_dict, title_maximum_length)
    body_indices = ConvertTokensListToIndices(bodyList, word_to_index_dict, body_maximum_length)
    train_feature_indices_Df = train_feature_indices_Df.append({'Title': title_indices, 'Body': body_indices}, ignore_index=True)
for row in validation_features_Df.itertuples():
    titleList = row[1]
    bodyList = row[2]
    title_indices = ConvertTokensListToIndices(titleList, word_to_index_dict, title_maximum_length)
    body_indices = ConvertTokensListToIndices(bodyList, word_to_index_dict, body_maximum_length)
    validation_feature_indices_Df = validation_feature_indices_Df.append({'Title': title_indices, 'Body': body_indices}, ignore_index=True)
    
train_feature_indices_Df.head()
from torch.utils.data import Dataset, DataLoader
import torch
useCuda = torch.cuda.is_available()

class ContextToWordDataset(Dataset):
    def __init__(self,
                 sentence_indices_dataframe,
                 context_length,
                 word_to_index_dict):
        self.sentence_indices_dataframe = sentence_indices_dataframe
        self.context_length = context_length
        self.word_to_index_dict = word_to_index_dict
        
    def __len__(self):
        return len(self.sentence_indices_dataframe)
    
    def __getitem__(self, idx):
        sentence_indices = self.sentence_indices_dataframe.iloc[idx]
        # Randomly select a target word
        last_acceptable_center_index = len(sentence_indices) - 1
        if self.word_to_index_dict['ENDOFSEQ'] in sentence_indices:
            for position, index in enumerate(sentence_indices):
                if index == self.word_to_index_dict['ENDOFSEQ']:
                    last_acceptable_center_index = position        
        targetNdx = random.choice(range(last_acceptable_center_index + 1))
        # Create a Long tensor with dim (2 * context_length)
        context_indicesTsr = torch.ones((2 * self.context_length)).long() * self.word_to_index_dict['NOTSET']
        runningNdx = targetNdx - int(self.context_length)
        counter = 0
        while counter < 2 * self.context_length:
            if runningNdx != targetNdx:
                if runningNdx >= 0 and runningNdx < len(sentence_indices):
                    context_indicesTsr[counter] = sentence_indices[runningNdx]
                counter += 1
            runningNdx += 1
        return (context_indicesTsr, torch.tensor(sentence_indices[targetNdx]).long())

context_length = 3
title_context_word_dataset = ContextToWordDataset(train_feature_indices_Df['Title'], context_length, word_to_index_dict)
class CenterWordPredictor(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension):
        super(CenterWordPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dimension)
        self.decoderLinear = torch.nn.Linear(embedding_dimension, vocabulary_size)

    def forward(self, contextTsr):
        # contextTsr.shape = (N, 2 * context_length); contextTsr.dtype = torch.int64
        embedding = self.embedding(contextTsr)  # (N, 2 * context_length, embedding_dimension)
        # Average over context words: (N, 2 * context_length, embedding_dimension) -> (N, embedding_dimension)
        embedding = torch.mean(embedding, dim=1)

        # Decoding
        outputTsr = self.decoderLinear(embedding)
        return outputTsr

embedding_dimension = 128
title_center_word_predictor = CenterWordPredictor(len(word_to_index_dict), embedding_dimension)
if useCuda:
    title_center_word_predictor = title_center_word_predictor.cuda()
def TrainCenterWordPredictor(predictor, optimizer, lossFcn, train_dataLoader, number_of_epochs):
    for epoch in range(1, number_of_epochs + 1):
        predictor.train()
        loss_sum = 0.0
        number_of_batches = 0
        for (context_indices_Tsr, target_word_Ndx_Tsr) in train_dataLoader:
            if number_of_batches % 10 == 1:
                print (".", end="", flush=True)
            if useCuda:
                context_indices_Tsr = context_indices_Tsr.cuda()
                target_word_Ndx_Tsr = target_word_Ndx_Tsr.cuda()
            predicted_center_word_ndx = predictor(context_indices_Tsr)
            optimizer.zero_grad()
            loss = lossFcn(predicted_center_word_ndx, target_word_Ndx_Tsr)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            number_of_batches += 1
        train_loss = loss_sum/number_of_batches
        print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))
        
word_predictor_parameters = filter(lambda p: p.requires_grad, title_center_word_predictor.parameters())
optimizer = torch.optim.Adam(word_predictor_parameters, lr=0.0001)
lossFcn = torch.nn.CrossEntropyLoss()
train_dataLoader = DataLoader(title_context_word_dataset, batch_size=32, shuffle=True)

TrainCenterWordPredictor(title_center_word_predictor, optimizer, lossFcn, train_dataLoader, 50)
body_context_word_dataset = ContextToWordDataset(train_feature_indices_Df['Body'], context_length, word_to_index_dict)
body_center_word_predictor = CenterWordPredictor(len(word_to_index_dict), embedding_dimension)
if useCuda:
    body_center_word_predictor = body_center_word_predictor.cuda()
word_predictor_parameters = filter(lambda p: p.requires_grad, body_center_word_predictor.parameters())
optimizer = torch.optim.Adam(word_predictor_parameters, lr=0.0001)
lossFcn = torch.nn.CrossEntropyLoss()
train_dataLoader = DataLoader(body_context_word_dataset, batch_size=32, shuffle=True)

TrainCenterWordPredictor(body_center_word_predictor, optimizer, lossFcn, train_dataLoader, 20)
class EmbeddingsToClassDataset(Dataset):
    def __init__(self,
                sentence_indices_dataframe,
                title_embedding,
                body_embedding,
                title_maximum_length,
                body_maximum_length,
                end_of_seq_Ndx,
                not_set_Ndx,
                target_class_dataframe,
                class_to_index_dict):
        self.sentence_indices_dataframe = sentence_indices_dataframe
        self.title_embedding = title_embedding
        self.body_embedding = body_embedding
        self.title_maximum_length = title_maximum_length
        self.body_maximum_length = body_maximum_length
        self.end_of_seq_Ndx = end_of_seq_Ndx
        self.not_set_Ndx = not_set_Ndx
        self.target_class_dataframe = target_class_dataframe
        self.class_to_index_dict = class_to_index_dict
        
    def __len__(self):
        return len(self.sentence_indices_dataframe)
    
    def __getitem__(self, idx):
        title_indices = self.sentence_indices_dataframe.iloc[idx]['Title']
        body_indices = self.sentence_indices_dataframe.iloc[idx]['Body']
        not_set_embedding = self.title_embedding.weight[self.not_set_Ndx]
        title_embedding_Tsr = torch.zeros(self.title_maximum_length, self.title_embedding.weight.shape[1])
        for rowNdx in range(title_embedding_Tsr.shape[0]):
            title_embedding_Tsr[rowNdx] = not_set_embedding
        end_of_seq_is_found = False
        runningNdx = 0
        while not end_of_seq_is_found and runningNdx < len(title_indices):
            wordNdx = title_indices[runningNdx]
            if wordNdx == self.end_of_seq_Ndx:
                end_of_seq_is_found = True
            word_embedding_Tsr = self.title_embedding.weight[wordNdx]
            title_embedding_Tsr[runningNdx] = word_embedding_Tsr
            runningNdx += 1
        
        not_set_embedding = self.body_embedding.weight[self.not_set_Ndx]
        body_embedding_Tsr = torch.zeros(self.body_maximum_length, self.body_embedding.weight.shape[-1]) 
        for rowNdx in range(body_embedding_Tsr.shape[0]):
            body_embedding_Tsr[rowNdx] = not_set_embedding
        end_of_seq_is_found = False
        runningNdx = 0
        while not end_of_seq_is_found and runningNdx < self.body_maximum_length:
            wordNdx = body_indices[runningNdx]
            if wordNdx == self.end_of_seq_Ndx:
                end_of_seq_is_found = True
            word_embedding_Tsr = self.body_embedding.weight[wordNdx]
            body_embedding_Tsr[runningNdx] = word_embedding_Tsr
            runningNdx += 1
            
        #print ("self.target_class_dataframe.iloc[idx] = {}".format(self.target_class_dataframe.iloc[idx]))
        target_class_index = self.class_to_index_dict[ self.target_class_dataframe.iloc[idx] ]
        return ((title_embedding_Tsr, body_embedding_Tsr), torch.tensor(target_class_index).long())
        
            
        
        
class_to_index_dict = {'HQ': 0, 'LQ_EDIT': 1, 'LQ_CLOSE': 2}
body_maximum_length = 100
training_embeddings_to_class_dataset = EmbeddingsToClassDataset(
                sentence_indices_dataframe=train_feature_indices_Df,
                title_embedding=title_center_word_predictor.embedding,
                body_embedding=body_center_word_predictor.embedding,
                title_maximum_length=title_maximum_length,
                body_maximum_length=body_maximum_length,
                end_of_seq_Ndx=word_to_index_dict['ENDOFSEQ'],
                not_set_Ndx=word_to_index_dict['NOTSET'],
                target_class_dataframe=train_target_Df,
                class_to_index_dict=class_to_index_dict)
validation_embeddings_to_class_dataset = EmbeddingsToClassDataset(
                sentence_indices_dataframe=validation_feature_indices_Df,
                title_embedding=title_center_word_predictor.embedding,
                body_embedding=body_center_word_predictor.embedding,
                title_maximum_length=title_maximum_length,
                body_maximum_length=body_maximum_length,
                end_of_seq_Ndx=word_to_index_dict['ENDOFSEQ'],
                not_set_Ndx=word_to_index_dict['NOTSET'],
                target_class_dataframe=validation_target_Df,
                class_to_index_dict=class_to_index_dict)
((title_embedding_Tsr, body_embeding_Tsr), target_class_Ndx_Tsr) = training_embeddings_to_class_dataset[0]
print ("title_embedding_Tsr.shape = {}".format(title_embedding_Tsr.shape))
print ("title_embedding_Tsr = {}".format(title_embedding_Tsr))
print ("body_embeding_Tsr.shape = {}".format(body_embeding_Tsr.shape))
print ("body_embeding_Tsr = {}".format(body_embeding_Tsr))
print ("target_class_Ndx_Tsr = {}".format(target_class_Ndx_Tsr))
class Double_LSTM(torch.nn.Module):
    def __init__(self, embedding_dimension, 
                 lstm_hidden_dimension,
                 num_lstm_layers, 
                 mlp_hidden_layer_dimension,
                 number_of_classes,
                 dropoutProportion=0.5):
        super(Double_LSTM, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.title_lstm = torch.nn.LSTM(embedding_dimension, lstm_hidden_dimension, num_lstm_layers,
                                  batch_first=True)
        self.body_lstm = torch.nn.LSTM(embedding_dimension, lstm_hidden_dimension, num_lstm_layers,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(dropoutProportion)
        self.linear1 = torch.nn.Linear(2 * lstm_hidden_dimension, number_of_classes) #mlp_hidden_layer_dimension)
        self.linear2 = torch.nn.Linear(mlp_hidden_layer_dimension, number_of_classes)
        
    def forward(self, inputTsr):
        # inputTsr[0].shape = (N, title_maximum_length, embedding_dimension)
        # inputTsr[1].shape = (N, body_maximum_length, embedding_dimension)
        title_embedding = inputTsr[0]
        body_embedding = inputTsr[1]
        title_aggregated_h, (title_ht, title_ct) = self.title_lstm(title_embedding)
        # title_ht.shape = (num_lstm_layers, N, lstm_hidden_dimension)
        # title_ht[-1].shape = (N, lstm_hidden_dimension)
        body_aggregated_h, (body_ht, body_ct) = self.body_lstm(body_embedding)
        # body_ht.shape = (num_lstm_layers, N, lstm_hidden_dimension)
        # body_ht[-1].shape = (N, lstm_hidden_dimension)
        concatenated_latent_Tsr = torch.cat((title_ht[-1], body_ht[-1]), dim=1)
        concatenated_latent_Tsr = self.dropout(concatenated_latent_Tsr)
        outputTsr = self.linear1(concatenated_latent_Tsr)
        
        """hidden_latent_Tsr = torch.nn.functional.relu(self.linear1(concatenated_latent_Tsr) )
        hidden_latent_Tsr = self.dropout(hidden_latent_Tsr)
        outputTsr = self.linear2(hidden_latent_Tsr)
        """

        return outputTsr
        
lstm_hidden_dimension = 64
lstm_number_of_layers = 1
mlp_hidden_dimension = 1024

double_lstm = Double_LSTM(
    embedding_dimension=embedding_dimension, 
    lstm_hidden_dimension=lstm_hidden_dimension,
    num_lstm_layers=lstm_number_of_layers, 
    mlp_hidden_layer_dimension=mlp_hidden_dimension,
    number_of_classes=3,
    dropoutProportion=0.5
)
if useCuda:
    double_lstm = double_lstm.cuda()
import sys

parameters = filter(lambda p: p.requires_grad, double_lstm.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0003)
lossFcn = torch.nn.CrossEntropyLoss()
train_dataLoader = DataLoader(training_embeddings_to_class_dataset, batch_size=32, shuffle=True)
validation_dataLoader = DataLoader(validation_embeddings_to_class_dataset, batch_size=32)
best_model_filepath = '/kaggle/working/double_lstm.pth'

lowestValidationLoss = sys.float_info.max
for epoch in range(0, 15 + 1):
    double_lstm.train()
    loss_sum = 0.0
    numberOfBatches = 0
    if epoch > 0:
        for (title_embedding_Tsr, body_embedding_Tsr), target_class_index in train_dataLoader:
            if numberOfBatches % 4 == 1:
                print (".", end="", flush=True)
            if useCuda:
                title_embedding_Tsr = title_embedding_Tsr.cuda()
                body_embedding_Tsr = body_embedding_Tsr.cuda()
                target_class_index =  target_class_index.cuda()
            predicted_index_Tsr = double_lstm((title_embedding_Tsr, body_embedding_Tsr))
            optimizer.zero_grad()
            loss = lossFcn(predicted_index_Tsr, target_class_index)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            numberOfBatches += 1
        train_loss = loss_sum/numberOfBatches
        print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))

    # Validation
    double_lstm.eval()
    with torch.no_grad():
        validation_loss_sum = 0
        validation_correct_predictions = 0
        number_of_validation_minibatches = 0
        for (validation_title_embedding_Tsr, validation_body_embedding_Tsr), validation_target_class_index in validation_dataLoader:
            if useCuda:
                validation_title_embedding_Tsr = validation_title_embedding_Tsr.cuda()
                validation_body_embedding_Tsr = validation_body_embedding_Tsr.cuda()
                validation_target_class_index = validation_target_class_index.cuda()
            validation_predicted_index_Tsr = double_lstm((validation_title_embedding_Tsr, validation_body_embedding_Tsr))
            validation_loss = lossFcn(validation_predicted_index_Tsr, validation_target_class_index).item()
            validation_loss_sum += validation_loss
            validation_correct_predictions += (validation_predicted_index_Tsr.argmax(dim=1) == validation_target_class_index).sum().item()
            number_of_validation_minibatches += 1
        # Validation accuracy      
        validation_accuracy = validation_correct_predictions / validation_embeddings_to_class_dataset.__len__()
        validation_loss = validation_loss_sum/number_of_validation_minibatches
        print ("validation_loss = {}; validation_accuracy = {}".format(validation_loss, validation_accuracy))

    if validation_loss < lowestValidationLoss:
        lowestValidationLoss = validation_loss
        torch.save(double_lstm.state_dict(), best_model_filepath)
