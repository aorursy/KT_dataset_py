!pip install pandas==0.24.0

!pip install tqdm==4.32.2

!pip install tqdm boto3 requests regex

!pip install pytorch_pretrained_bert pytorch-nlp
%matplotlib inline

import sys

import itertools

import numpy as np

import random as rn

import matplotlib.pyplot as plt

import torch

from pytorch_pretrained_bert import BertModel

from torch import nn

from pytorch_pretrained_bert import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch.optim import Adam

from torch.nn.utils import clip_grad_norm_

from IPython.display import clear_output

from sklearn.model_selection import train_test_split



from tqdm import tqdm

import pandas as pd

tqdm.pandas()



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
dataset = pd.read_json('/kaggle/input/News_Category_Dataset_v2.json', lines=True)
dataset.head()
cats = dataset.groupby('category')

print(cats.size(), "\n total groups: ",cats.ngroups)
# merge categories which should be the same

dataset.category.replace('THE WORLDPOST', 'WORLDPOST', inplace=True)

dataset.category.replace('WORLDPOST', 'WORLD NEWS', inplace=True)

dataset.category.replace('ARTS', 'ARTS & CULTURE', inplace=True)

dataset.category.replace('CULTURE & ARTS', 'ARTS & CULTURE', inplace=True)

dataset.category.replace('PARENTS', 'PARENTING', inplace=True)

dataset.category.replace('STYLE', 'STYLE & BEAUTY', inplace=True)
# headline plus short description

dataset['contents'] = dataset.headline + ". " + dataset.short_description
train, test = train_test_split(dataset, test_size=0.2)
train.head()
train_contents, train_cats = train['contents'] , train['category'] # select the test and training data

test_contents, test_cats = test['contents'] , test['category']

len(train_contents), len(train_cats), len(test_contents), len(test_cats)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # initialise the BERT tokenizer
MAX_LEN = 128
train_tokens = train_contents.progress_apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x)[:MAX_LEN] + ['[SEP]']) # BERT accepts a [CLS] token as the start of a sentence, so for each dataset add this and clip it at max 511 lenth. [SEP] should be at the end?

test_tokens = test_contents.progress_apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x)[:MAX_LEN] + ['[SEP]'])
train_tokens.iloc[0]
train_tokens_ids = train_tokens.progress_apply(tokenizer.convert_tokens_to_ids) # convert the tokens to ids

test_tokens_ids = test_tokens.progress_apply(tokenizer.convert_tokens_to_ids)
train_tokens_ids
train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=MAX_LEN, truncating="post", padding="post", dtype="int") # pad sequences

test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=MAX_LEN, truncating="post", padding="post", dtype="int")
train_tokens_ids # they're now numpy arrays
train_tokens_ids.shape, test_tokens_ids.shape
train_y = train_cats.astype('category').cat.codes #  convert each categorical label into a number to label the category

test_y = test_cats.astype('category').cat.codes
train_y = train_y.astype('int64').values

test_y = test_y.astype('int64').values
#  now convert these to one hot encoded!

#no_labels = len(train_cats.astype('category').cat.categories)

#train_y = np.eye(no_labels)[train_y] # creates onehot array

#test_y = np.eye(no_labels)[test_y]
train_y.shape, test_y.shape
train_masks = (train_tokens_ids > 0).astype('float')

train_masks = train_masks.tolist()



test_masks = (test_tokens_ids > 0).astype('float')

test_masks = test_masks.tolist()
train_masks
baseline_model = make_pipeline(CountVectorizer(ngram_range=(1,3)), SGDClassifier(loss='log', verbose=2, n_jobs=-1, max_iter=10, learning_rate='optimal', n_iter_no_change=5, validation_fraction=0.1, alpha=0.001)) # use sgd as we have a large model. log loss makes it logistic regression
baseline_model = baseline_model.fit(train_contents, train_cats)
import pickle

pkl_filename = "../working/news_logistic_bigsgd_higherreg.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(baseline_model, file)
import pickle

pkl_filename = "../working/news_logistic_bigsgd_higherreg.pkl"

pickle.load(open(pkl_filename, "rb"))
baseline_predicted = baseline_model.predict(test_contents) # predict the categories for the test set
print(classification_report(test_cats, baseline_predicted)) # report the scores
rn.seed(42)

np.random.seed(42)

torch.manual_seed(42)

torch.cuda.manual_seed(42)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
from pytorch_pretrained_bert import BertForSequenceClassification
no_labels = len(train_cats.astype('category').cat.categories)

bert_clf = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=no_labels)

bert_clf = bert_clf.cuda()
BATCH_SIZE = 64

EPOCHS = 4
param_optimizer = list(bert_clf.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]
from pytorch_pretrained_bert import BertAdam

optimizer = BertAdam(optimizer_grouped_parameters,

                     lr=3e-5,

                     warmup=.1)
# convert to tensors

train_tokens_tensor = torch.tensor(train_tokens_ids)

train_y_tensor = torch.tensor(train_y, dtype=torch.long)



test_tokens_tensor = torch.tensor(test_tokens_ids)

test_y_tensor = torch.tensor(test_y, dtype=torch.long)



train_masks_tensor = torch.tensor(train_masks,  dtype=torch.long)

test_masks_tensor = torch.tensor(test_masks,  dtype=torch.long)



str(torch.cuda.memory_allocated(device)/1000000 ) + 'M'
# convert a;; to dataloader to save memory later

train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)

train_sampler = RandomSampler(train_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)



test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)

test_sampler = SequentialSampler(test_dataset)

test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
loss_func = nn.BCEWithLogitsLoss().cuda()
# todo:

# split data into validation set also?

# add code for validation evalutation?

# clean up this notebook a bit.

# evaluate the classifier

# 
token_ids.shape, masks.shape, labels.shape
losses = []

steps = []

step = 0

for epoch_num in range(EPOCHS):

    bert_clf.train()

    train_loss = 0

    for step_num, batch_data in enumerate(train_dataloader): # train for each epoch

        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        optimizer.zero_grad() # clear gradients

        loss =  bert_clf(input_ids=token_ids, attention_mask=masks, labels=labels) # forward pass

        losses.append(loss.item())

        loss.backward() # backward pass

        #import ipdb

        #ipdb.set_trace()



        clip_grad_norm_(parameters=bert_clf.parameters(), max_norm=1.0)

        

        optimizer.step() # update params and step using computed gradient

        

        clear_output(wait=True)

        

        train_loss += loss.item()

        steps.append(step)

        step += 1

        print('Epoch: ', epoch_num + 1)

        print("{0}/{1} loss: {2} ".format(step_num, len(train) / BATCH_SIZE, train_loss / (step_num + 1)))

        

    torch.save({

            'epoch': epoch_num,

            'model_state_dict': bert_clf.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'loss': loss

            }, "../working/news_bert_chpt.pt")
PATH = "../working/news_bert_statedict.pt"

toch.save(bert_clf.state_dict(), PATH)
PATH = "../working/news_bert.pt"

toch.save(bert_clf, PATH)
bert_clf.eval() # put model in evaluation mode

bert_predicted = []

with torch.no_grad():

    for step_num, batch_data in enumerate(test_dataloader):



        token_ids, masks = tuple(t.to(device) for t in batch_data)

        logits_predictions =  bert_clf(input_ids=token_ids, attention_mask=masks) # forward pass

        

        logits = logits_predictions.detach().cpu().numpy() #move to cpu

        

        bert_predicted += (np.argmax(logits, axis=1)).tolist()
print(classification_report(test_y, bert_predicted))