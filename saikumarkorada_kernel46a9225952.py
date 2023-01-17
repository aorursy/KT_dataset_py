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
from __future__ import absolute_import, division



import os

import time

import numpy as np

import pandas as pd

import gensim

from tqdm import tqdm

from nltk.stem import PorterStemmer

ps = PorterStemmer()

from nltk.stem.lancaster import LancasterStemmer

lc = LancasterStemmer()

from nltk.stem import SnowballStemmer

sb = SnowballStemmer("english")

import gc
start_time = time.time()

print("Loading data ...")

train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv').fillna(' ')

test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv').fillna(' ')

train_texts = train['question_text']

test_texts = test['question_text']

text_list = pd.concat([train_texts, test_texts])

train_labels=train['question_text']

y = train['target'].values

num_train_data = y.shape[0]

print("--- %s seconds ---" % (time.time() - start_time))
!pip install pytorch-pretrained-bert
# BERT imports

import torch 

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertAdam,BertModel, BertForSequenceClassification

from tqdm import tqdm, trange

import pandas as pd

import io

import numpy as np

import matplotlib.pyplot as plt





tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], train_texts))

test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:511], test_texts))
train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")

test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=128, truncating="post", padding="post", dtype="int")
#train_y = np.array(y) == 'pos'

train_y=y.copy()
test_y=np.zeros(test.shape[0])
class BertBinaryClassifier(nn.Module):

    def __init__(self, dropout=0.1):

        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Linear(768, 1)

        self.sigmoid = nn.Sigmoid()

        #self.logsoftmax = nn.LogSoftmax(dim=1)

        #self.softmax = nn.Softmax(dim=1)

    

    def forward(self, tokens):

        _, pooled_output = self.bert(tokens, output_all_encoded_layers=False)

        linear_output = self.linear(pooled_output)

        proba = self.sigmoid(linear_output)

        #proba=self.logsoftmax(linear_output)

        #proba = self.softmax(linear_output)

        return proba

        #return linear_output

        

    def freeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = False

    

    def unfreeze_bert_encoder(self):

        for param in self.bert.parameters():

            param.requires_grad = True
BATCH_SIZE=32

train_tokens_tensor = torch.tensor(train_tokens_ids)

train_y_tensor = torch.tensor(train_y.reshape(-1, 1)).float()

test_tokens_tensor = torch.tensor(test_tokens_ids)

test_y_tensor = torch.tensor(test_y.reshape(-1, 1)).float()

train_dataset = TensorDataset(train_tokens_tensor, train_y_tensor)

train_sampler = RandomSampler(train_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_tokens_tensor, test_y_tensor)

test_sampler = SequentialSampler(test_dataset)

test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
device = torch.device("cuda")

n_gpu = torch.cuda.device_count()
n_gpu
start_time=time.time()

EPOCHS=4

bert_clf = BertBinaryClassifier()

bert_clf.to(device)

bert_clf.freeze_bert_encoder()

bert_clf = bert_clf.cuda()

optimizer = BertAdam(bert_clf.parameters(), lr=3e-6) 

bert_clf.train()



for epoch_num in range(EPOCHS):

    for step_num, batch_data in enumerate(train_dataloader):

        token_ids, labels = tuple(t.to(device) for t   in batch_data)

        probas = bert_clf(token_ids)

        #_, probas = torch.max(probas, 1)

        loss_func = nn.BCELoss()

        

        batch_loss = loss_func(probas, labels)

        bert_clf.freeze_bert_encoder()

        bert_clf.zero_grad()

        batch_loss.backward() 

        optimizer.step()



print("--- %s seconds ---" % (time.time() - start_time))
torch.save({

            'epoch': epoch_num,

            'model_state_dict': bert_clf.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'loss': batch_loss,

            }, "Quora_Bert_Pytorch.pth")


bert_clf.eval()

res = []

for step_num, batch_data in enumerate(test_dataloader):

    token_ids,labels = tuple(t.to(device) for t in batch_data)

    with torch.no_grad():

        logits = bert_clf(token_ids)

    logits = logits.detach().cpu().numpy()

    for t in logits:

        if t>0.5:

            res.append(1)

        else:

            res.append(0)

#　For Submission



test['prediction'] = res





submission = test[["qid","prediction"]]



#submission.columns = ['Id', 'Category']

submission.to_csv('submission.csv', index=False)

submission.head()