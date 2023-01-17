# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pytorch_pretrained_bert

!pip install pytorch-nlp
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer, BertModel

import torch

from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report

import random
# READ THE TRAINING AND TESTING DATASET

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')



train_data
# Ramdonly shuffle the dataset

train_data = train_data.sample(frac=1, random_state=24).reset_index(drop=True)

train_data.head()

train_data = train_data.drop(columns = ['id','keyword', 'location'])

# WE DON'T FOLLOW THE PREVIOUS LINK

# DATA PREPROCESSING

# Retain only alphabets

train_data['text'] = train_data['text'].str.replace("[^a-zA-Z]", " ")

train_data
# Get rid of stopwords

from nltk.corpus import stopwords 

stop_words = stopwords.words('english')



#Tokenization

tokenized_doc = train_data['text'].apply(lambda x: x.split())

tokenized_doc



# Remove stop words

tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

tokenized_doc



# De-tokenization

detokenized_doc = []

for i in range(len(train_data)):

    t = ' '.join(tokenized_doc[i])

    detokenized_doc.append(t)

    

train_data['text'] = detokenized_doc

train_data
# SPLIT THE DATASET INTO TRAINING(60%) AND VALIDATION(40%) SETS

from sklearn.model_selection import train_test_split

train_data, valid_data = train_test_split(train_data, test_size=0.4, random_state=12)

valid_data.shape
train_dataset = train_data.to_dict('records')

valid_dataset = valid_data.to_dict('records')

train_dataset[:5]
# List of dictionaries with 'text' and 'target' keys

#train_dataset1 = [{'target':target_data, 'text':text} for target_data in list(train_data['target']) for text in list(train_data['text'])]

#train_dataset2 = [{'text':text} for text in list(train_data['text'])]

#valid_dataset = [{'text':text, 'target':target_data} for text in list(train_data['text']) for target_data in list (valid_data['target'])]

#train_dataset = {**train_dataset1, **train_dataset2}
# Generate a list of tuples

train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['target']), train_dataset)))

valid_texts, valid_labels = list(zip(*map(lambda d: (d['text'], d['target']), valid_dataset)))

train_texts[:7]

# Generate tokens and token ids.

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)

train_tokens = list(map(lambda t:['[CLS]'] + tokenizer.tokenize(t)[:127], train_texts))

valid_tokens = list(map(lambda t:['[CLS]'] + tokenizer.tokenize(t)[:127], valid_texts))

train_tokens[:2]
# Generate tokens and token ids.

# We truncate the input strings to 128 characteres --> less costly

train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

valid_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, valid_tokens))



train_tokens_ids = pad_sequences (train_tokens_ids, maxlen=128, truncating = "post", padding = "post", dtype = "int")

valid_tokens_ids = pad_sequences (valid_tokens_ids, maxlen=128, truncating = "post", padding = "post", dtype = "int")

len(train_tokens_ids)
# Boolean array based on the value of target 

train_y = np.array(train_labels) == 1

valid_y = np.array(valid_labels) == 1
class BertBinaryClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertBinaryClassifier, self).__init__()



        self.bert = BertModel.from_pretrained('bert-base-uncased')



        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(768, 1)

        self.sigmoid = nn.Sigmoid()

    

    def forward(self, tokens, masks=None):

        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)

        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)

        proba = self.sigmoid(linear_output)

        return proba
BATCH_SIZE = 32

EPOCHS = 3
# Generate the masks

train_masks = [[float(i>0) for i in ii] for ii in train_tokens_ids] #gives 1 and 0s

valid_masks = [[float(i>0) for i in ii] for ii in valid_tokens_ids] 

train_masks_tensor = torch.tensor(train_masks)

valid_masks_tensor = torch.tensor(valid_masks)
# Generate token tensors 

train_tokens_tensor = torch.tensor(train_tokens_ids)

train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()

valid_tokens_tensor = torch.tensor(valid_tokens_ids)

valid_y_tensor = torch.tensor(valid_y.reshape(-1, 1)).float()

train_dataset =  torch.utils.data.TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)

train_sampler =  torch.utils.data.RandomSampler(train_dataset)

train_dataloader =  torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset =  torch.utils.data.TensorDataset(valid_tokens_tensor, valid_masks_tensor, valid_y_tensor)

valid_sampler =  torch.utils.data.SequentialSampler(valid_dataset)

valid_dataloader =  torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

# Prepare our data loaders





# Adam optimizer is used to minimize the Binary Cross Entropy Loss(BCELoss) and trained with a batch size of 1 for 1 epoch.

bert_clf = BertBinaryClassifier()

optimizer = torch.optim.Adam(bert_clf.parameters(), lr=3e-6)

for epoch_num in range(EPOCHS):

    bert_clf.train()

    train_loss = 0

    for step_num, batch_data in enumerate(train_dataloader):

        token_ids, masks, labels = tuple(t for t in batch_data)

        probas = bert_clf(token_ids, masks)

        loss_func = nn.BCELoss()

        batch_loss = loss_func(probas, labels)

        train_loss += batch_loss.item()

        bert_clf.zero_grad()

        batch_loss.backward()

        optimizer.step()

        print('Epoch: ', epoch_num + 1)

        print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_data) / BATCH_SIZE, train_loss / (step_num + 1)))
bert_clf.eval()

bert_predicted = []

all_logits = []

with torch.no_grad():

    for step_num, batch_data in enumerate(valid_dataloader):



        token_ids, masks, labels = tuple(t for t in batch_data)



        logits = bert_clf(token_ids, masks)

        loss_func = nn.BCELoss()

        loss = loss_func(logits, labels)

        numpy_logits = logits.cpu().detach().numpy()

        

        bert_predicted += list(numpy_logits[:, 0] > 0.5)

        all_logits += list(numpy_logits[:, 0])

     

print(classification_report(valid_y, bert_predicted))
len(bert_predicted)
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

#test_data = test_data.drop(columns = ['keyword','location'])

test_data = test_data.drop(columns = ['keyword', 'location'])



# WE DON'T FOLLOW THE PREVIOUS LINK

# DATA PREPROCESSING

# Retain only alphabets

test_data['text'] = test_data['text'].str.replace("[^a-zA-Z]", " ")

test_data



# Get rid of stopwords

from nltk.corpus import stopwords 

stop_words = stopwords.words('english')



#Tokenization

tokenized_doc = test_data['text'].apply(lambda x: x.split())

tokenized_doc



# Remove stop words

tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

tokenized_doc



# De-tokenization

detokenized_doc = []

for i in range(len(test_data)):

    t = ' '.join(tokenized_doc[i])

    detokenized_doc.append(t)

    

test_data['text'] = detokenized_doc

test_data
test_dataset = test_data.to_dict('records')



test_dataset[:5]
test_texts, _= list(zip(*map(lambda d: (d['text'], _), test_dataset)))

test_texts[:5]
test_tokens = list(map(lambda t:['[CLS]'] + tokenizer.tokenize(t)[:127], test_texts))

test_tokens[:2]
test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

test_tokens_ids[:4]
# We have to pad

test_tokens_ids = pad_sequences (test_tokens_ids, maxlen=128, truncating = "post", padding = "post", dtype = "int")

test_tokens_ids[:1]

test_masks = [[float(i>0) for i in ii] for ii in test_tokens_ids] 
test_tokens_tensor = torch.tensor(test_tokens_ids) #This won’t work, as your input has varying shapes in dim1. You could pad the last row with some values: a = [[1,2,3],[4,5,6],[1, 0, 0]] b = torch.tensor(a) 

test_masks_tensor = torch.tensor(test_masks)
test_dataset =  torch.utils.data.TensorDataset(test_tokens_tensor, test_masks_tensor)

test_sampler =  torch.utils.data.SequentialSampler(test_dataset)

test_dataloader =  torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
bert_clf.eval()

bert_predicted = []

all_logits = []


with torch.no_grad():

    for step_num, batch_data in enumerate(test_dataloader):



        token_ids, masks = tuple(t for t in batch_data)



        logits = bert_clf(token_ids, masks)



        numpy_logits = logits.cpu().detach().numpy()

        

        bert_predicted += list(numpy_logits[:, 0] > 0.5)

        all_logits += list(numpy_logits[:, 0])



len(bert_predicted)
test_data['target'] = bert_predicted

test_data.target = test_data.target.astype(int)

test_data = test_data.drop(columns = ['text'])



test_data
test_data.to_csv('submission.csv', index=False)