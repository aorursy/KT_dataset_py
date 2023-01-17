import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

print(sys.executable)
!nvidia-smi
# uncomment/Comment below line to install/skip-install hugging-face transformers



!pip install transformers
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb # pytorch-transformers by huggingface

import time

import warnings

warnings.filterwarnings('ignore')
#initiating Garbage Collector for GPU environment setup

import gc

for obj in gc.get_objects():

    try:

        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):

            print(type(obj), obj.size())

    except:

        pass
torch.cuda.is_available()
path = '../input/stanford-sentiment-treebank-v2-sst2/datasets/'



# to read via CSV files...

# df = pd.read_csv(path + 'csv-format/train.csv')



df = pd.read_csv(path + 'tsv-format/train.tsv', delimiter='\t')
df.shape
batch_1 = df[:2000]

batch_1['Ratings'].value_counts()
# https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu

USE_GPU = True



if USE_GPU and torch.cuda.is_available():

    print('using device: cuda')

else:

    print('using device: cpu')
use_cuda = not False and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print(time.ctime())





model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')



## Want BERT instead of distilBERT? Uncomment the following line:

#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')



# Load pretrained model/tokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights).to(device)



print(time.ctime())
tokenized = batch_1['Reviews'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokenized.shape
print(time.ctime())



max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])



print(time.ctime())
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
# with GPU usage...





print(time.ctime())





if USE_GPU and torch.cuda.is_available():

    print('using GPU...')

    input_ids = torch.tensor(padded).to(device)  

    attention_mask = torch.tensor(attention_mask).to(device)



    with torch.no_grad():

        last_hidden_states = model(input_ids, attention_mask=attention_mask)# .to(device)

        

print(time.ctime())
# add .cpu to convert cuda tensor to numpy()



features = last_hidden_states[0][:,0,:].cpu().numpy()
labels = batch_1['Ratings']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


# parameters = {'C': np.linspace(0.0001, 100, 20)}

# grid_search = GridSearchCV(LogisticRegression(), parameters)

# grid_search.fit(train_features, train_labels)



# print('best parameters: ', grid_search.best_params_)

# print('best scrores: ', grid_search.best_score_)
lr_clf = LogisticRegression()

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)
from sklearn.dummy import DummyClassifier

clf = DummyClassifier()



scores = cross_val_score(clf, train_features, train_labels)

print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))