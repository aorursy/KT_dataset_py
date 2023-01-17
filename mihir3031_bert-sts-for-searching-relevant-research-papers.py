# importing important libraries for code

import csv
import os
import random
import sys

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import unidecode
import re

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
from scipy.stats.stats import pearsonr

from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertModel
# Checking for GPU availability

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
n_gpu = torch.cuda.device_count()
print("device: {} n_gpu: {}".format(device, n_gpu))
# Use this class for linear regression model

class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(768, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out
# Initializing the fine-tuned model

model_class= BertModel
model_dir= '../input/models/'

# Load a trained model and config that you have fine-tuned
tokenizer = BertTokenizer.from_pretrained('../input/tokenizer/bert-base-uncased-vocab.txt')
model = model_class.from_pretrained(model_dir)
regression = torch.load(join(model_dir,"regression_model.pth"), map_location=torch.device(device))
model.to(device)
regression.to(device)

# Initializing the parameters
max_seq_length= 128
batch_size=1
# You can update this query list for your desire output

query_list = ['vaccine vaccination dose antitoxin serum immunization inoculation for covid 19 or coronavirus related research work', 
              'therapeutics treatment therapy drug antidotes cures remedies medication prophylactic restorative panacea for covid 19 or coronavirus']
#Initialize evaluation mode
model.eval()
regression.eval()

#Loading the test set given by this challange
test_path= '../input/CORD-19-research-challenge/metadata.csv'
df_test= pd.read_csv(test_path)

#Empty List for saving the final computated score
final_score= list()
query_number=1

# Loop that calculate score corresponding to each query given in above list
for query in query_list:
    for i in range(len(df_test)):
        
        # Aggregating the [title + abstract + journal]. You can modify it according to your input
        document= str(df_test.iloc[i].title) + str(df_test.iloc[i].abstract) + str(df_test.iloc[i].journal)
    
        iteration = int(len(document)/max_seq_length)
    
        if iteration==0:
            final_score.append(0)
            continue
    
        result= list()
        sentence= list()
        
        # Loop for create sentence inputs
        for i in range(0,iteration):
            sent= document[(i*max_seq_length):(i+1)*max_seq_length]
            sentence.append(sent)
        
        df_temp= pd.DataFrame(sentence, columns=['d_sent'])
        df_temp['q']= query
    
        # Create text sequence
        sentences_1 = df_temp.q
        sentences_2 = df_temp.d_sent
    
        # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
        special_sentences_tempe_1 = ["[CLS] " + sentence for sentence in sentences_1]
        special_sentences_tempe_2 = [" [SEP] " + sentence for sentence in sentences_2]
        special_sentences = [i + j for i, j in zip(special_sentences_tempe_1, special_sentences_tempe_2)]
        
        tokenized_texts = [tokenizer.tokenize(sentence) for sentence in special_sentences]
        
        # Max sentence input 
        MAX_LEN = max_seq_length
        
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_sentences = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        
        # Pad our input tokens
        input_sentences = pad_sequences(input_sentences, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        
        # Create attention Masks
        attention_masks = []
        
        # Create a mask of 1s for each token followed by 0s for padding
        
        for seq in input_sentences:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        
        # Convert all of our data into torch tensors, the required datatype for our model
        test_inputs = torch.tensor(input_sentences)
        test_masks = torch.tensor(attention_masks)
        
        # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
        batch_size = batch_size
        
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory
        test_data = TensorDataset(test_inputs, test_masks)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        
        # Loop for calculating the similarity score corresponding to each sentence
        for step, batch in enumerate(test_dataloader):
            input_ids, input_mask= batch
            input_ids=input_ids.to(device)
            input_mask=input_mask.to(device)
            
            outputs = model(input_ids, attention_mask=input_mask)
            last_hidden_states = outputs[1]
            pred_score= regression(last_hidden_states)
            pred_score= np.squeeze(pred_score, axis=1)
            pred_score = pred_score.detach().cpu().numpy()
            result.extend(pred_score)
            
        result.sort(reverse=True)
        
        # This is to calculate final score. Here, we are using max sentence score only. You can change it according to your requirement.
        
        final_score.append(max(result)/5)
    
    result_query= 'q'+str(query_number)+'_score'
    query_number+=1
    df_test[result_query]= final_score

# Saving the final results as output.csv file
df_test.to_csv('../input/output/output.csv')
path= '../input/output/output.csv'
df= pd.read_csv(path)

label= list()

for i in range(len(df)):
    other_prob= 1 - df.iloc[i].q1_score - df.iloc[i].q2_score
    
    if other_prob<0:
        other.append(0)
    else:
        other.append(other_prob)
    
    max_val= max(df.iloc[i].q1_score, df.iloc[i].q2_score, other_prob)
    
    if max_val==df.iloc[i].vaccine:
        l='vaccine'
    elif max_val==df.iloc[i].therapeutics:
        l='therapeutics'
    elif max_val==other_prob:
        l='other'
    
    label.append(l)

df['label']= label
df = pd.read_csv("../input/output/final_data.csv")
df.head(40) # prints the first 40 rows of the table.