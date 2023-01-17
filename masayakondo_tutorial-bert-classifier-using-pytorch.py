# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from sklearn.model_selection import train_test_split





datasets = pd.read_csv('../input/nlp-getting-started/train.csv')

datasets.head()



# Make only the columns you need.

datasets = datasets[['text', 'target']]



# shuffle data

datasets = datasets.sample(frac=1).reset_index(drop=True)



train_df, test_df = train_test_split(datasets, train_size=0.95)

print('train data', train_df.shape)

print('test data', test_df.shape)

print(train_df.head())



# save as tsv file

train_df.to_csv('./train.tsv', sep='\t', index=False, header=None)

test_df.to_csv('./test.tsv', sep='\t', index=False, header=None)
import seaborn as sns

sns.distplot(datasets['target'])
import torch

from torch import nn

import torch.nn.functional as F

import torchtext

from transformers import BertTokenizer

from transformers.modeling_bert import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')



# Check Bert Network

print(model)
text = list(train_df['text'])[0]

print("sample text: ", text)

tokens = tokenizer.encode(text, return_tensors='pt')

print(tokens)

print(tokenizer.convert_ids_to_tokens(tokens[0].tolist()))
length = datasets['text'].map(tokenizer.encode).map(len)

print(max(length))

sns.distplot(length)

#The maximum length of the series that BERT can handle is 512, but the data in this case is fine.
def bert_tokenizer(text):

    return tokenizer.encode(text, return_tensors='pt')[0]



TEXT = torchtext.data.Field(sequential=True, tokenize=bert_tokenizer, use_vocab=False, lower=False,

                            include_lengths=True, batch_first=True, pad_token=0)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)



train_data, test_data = torchtext.data.TabularDataset.splits(

    path='./', train='train.tsv', test='test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])



BATCH_SIZE = 32

train_iter, test_iter = torchtext.data.Iterator.splits((train_data, test_data),

                                                       batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)
batch = next(iter(train_iter))

print(batch.Text)

print(batch.Label)
class BertClassifier(nn.Module):

    def __init__(self):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # BERT hidden state size is 768, class number is 2

        self.linear = nn.Linear(768, 2)

        # initialing weights and bias

        nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, input_ids):

        # get last_hidden_state

        vec, _ = self.bert(input_ids)

        # only get first token 'cls'

        vec = vec[:,0,:]

        vec = vec.view(-1, 768)



        out = self.linear(vec)

        return F.log_softmax(out)



classifier = BertClassifier()
# First, turn off the gradient for all parameters.

for param in classifier.parameters():

    param.requires_grad = False



# Second, turn on only last BERT layer.

for param in classifier.bert.encoder.layer[-1].parameters():

    param.requires_grad = True



# Finally, turn on classifier layer.

for param in classifier.linear.parameters():

    param.requires_grad = True



import torch.optim as optim



# The pre-learned sections should have a smaller learning rate, and the last total combined layer should be larger.

optimizer = optim.Adam([

    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},

    {'params': classifier.linear.parameters(), 'lr': 1e-4}

])



# loss function

loss_function = nn.NLLLoss()
# set GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# send network to GPU

classifier.to(device)

losses = []



for epoch in range(10):

    all_loss = 0

    for idx, batch in enumerate(train_iter):

        batch_loss = 0

        classifier.zero_grad()

        input_ids = batch.Text[0].to(device)

        label_ids = batch.Label.to(device)

        out = classifier(input_ids)

        batch_loss = loss_function(out, label_ids)

        batch_loss.backward()

        optimizer.step()

        all_loss += batch_loss.item()

    print("epoch", epoch, "\t" , "loss", all_loss)
from sklearn.metrics import classification_report



answer = []

prediction = []

with torch.no_grad():

    for batch in test_iter:



        text_tensor = batch.Text[0].to(device)

        label_tensor = batch.Label.to(device)



        score = classifier(text_tensor)

        _, pred = torch.max(score, 1)



        prediction += list(pred.cpu().numpy())

        answer += list(label_tensor.cpu().numpy())

print(classification_report(prediction, answer))
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

ids = list(test_df['id'])

test_df = test_df[['text']]

test_df['target'] = [0 for _ in range(test_df.shape[0])]

test_df.head()

test_df.to_csv('./s_test.tsv', sep='\t', index=False, header=None)



train_data, test_data = torchtext.data.TabularDataset.splits(

    path='./', train='train.tsv', test='s_test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])



BATCH_SIZE = 32

_, test_iter = torchtext.data.Iterator.splits((train_data, test_data),

                                                       batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)
prediction = []

with torch.no_grad():

    for batch in test_iter:



        text_tensor = batch.Text[0].to(device)

        label_tensor = batch.Label.to(device)



        score = classifier(text_tensor)

        _, pred = torch.max(score, 1)



        prediction += list(pred.cpu().numpy())
sub_dict = {'id':ids, 'target':prediction}



sub = pd.DataFrame.from_dict(sub_dict)

sub.to_csv("submission.csv", index = False)
from transformers import * 



roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

roberta_model = RobertaModel.from_pretrained('roberta-base')
def roberta_tokenizer(text):

    return tokenizer.encode(text, return_tensors='pt')[0]



TEXT = torchtext.data.Field(sequential=True, tokenize=roberta_tokenizer, use_vocab=False, lower=False,

                            include_lengths=True, batch_first=True, pad_token=0)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)



train_data, test_data = torchtext.data.TabularDataset.splits(

    path='./', train='train.tsv', test='test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])



BATCH_SIZE = 32

train_iter, test_iter = torchtext.data.Iterator.splits((train_data, test_data),

                                                       batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)
class RobertaClassifier(nn.Module):

    def __init__(self):

        super(RobertaClassifier, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # ROBERTA hidden state size is 768, class number is 2

        self.linear = nn.Linear(768, 2)

        # initialing weights and bias

        nn.init.normal_(self.linear.weight, std=0.02)

        nn.init.normal_(self.linear.bias, 0)



    def forward(self, input_ids):

        # get last_hidden_state

        vec, _ = self.roberta(input_ids)

        # only get first token 'cls'

        vec = vec[:,0,:]

        vec = vec.view(-1, 768)



        out = self.linear(vec)

        return F.log_softmax(out)



classifier = RobertaClassifier()
# First, turn off the gradient for all parameters.

for param in classifier.parameters():

    param.requires_grad = False



# Second, turn on only last BERT layer.

for param in classifier.roberta.encoder.layer[-1].parameters():

    param.requires_grad = True



# Finally, turn on classifier layer.

for param in classifier.linear.parameters():

    param.requires_grad = True



import torch.optim as optim



# The pre-learned sections should have a smaller learning rate, and the last total combined layer should be larger.

optimizer = optim.Adam([

    {'params': classifier.roberta.encoder.layer[-1].parameters(), 'lr': 5e-5},

    {'params': classifier.linear.parameters(), 'lr': 1e-4}

])



# loss function

loss_function = nn.NLLLoss()
# set GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# send network to GPU

classifier.to(device)

losses = []



for epoch in range(10):

    all_loss = 0

    for idx, batch in enumerate(train_iter):

        batch_loss = 0

        classifier.zero_grad()

        input_ids = batch.Text[0].to(device)

        label_ids = batch.Label.to(device)

        out = classifier(input_ids)

        batch_loss = loss_function(out, label_ids)

        batch_loss.backward()

        optimizer.step()

        all_loss += batch_loss.item()

    print("epoch", epoch, "\t" , "loss", all_loss)
answer = []

prediction = []

with torch.no_grad():

    for batch in test_iter:



        text_tensor = batch.Text[0].to(device)

        label_tensor = batch.Label.to(device)



        score = classifier(text_tensor)

        _, pred = torch.max(score, 1)



        prediction += list(pred.cpu().numpy())

        answer += list(label_tensor.cpu().numpy())

print(classification_report(prediction, answer))