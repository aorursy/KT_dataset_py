#!pip3 install fastai==1.0.42
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.text import * 

from fastai.gen_doc.nbdoc import *

from fastai.datasets import * 

from fastai.datasets import Config

from pathlib import Path

import pandas as pd
#import fastai; 

#fastai.show_install(1)
path = Path('../input/')
df = pd.read_csv(path/'train.tsv', sep="\t")

df.head()
df.shape
df['Sentiment'].value_counts()
df_test = pd.read_csv(path/'test.tsv', sep="\t")

df_test.head()
df = pd.DataFrame({'label':df.Sentiment, 'text':df.Phrase})

df.head()
test_df = pd.DataFrame({'PhraseId':df_test.PhraseId, 'text':df_test.Phrase})

test_df.head()
df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")

df_test['text'] = test_df['text'].str.replace("[^a-zA-Z]", " ")
import nltk

nltk.download('stopwords') 

from nltk.corpus import stopwords 
stop_words = stopwords.words('english')
# tokenization 

tokenized_doc = df['text'].apply(lambda x: x.split()) 
# remove stop-words 

tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 

                                    item not in stop_words]) 
# de-tokenization 

detokenized_doc = [] 
for i in range(len(df)):

    t =' '.join(tokenized_doc[i]) 

    detokenized_doc.append(t) 
df['text'] = detokenized_doc

df.head()
# de-tokenization 

detokenized_doc = [] 
df_test.head()
# tokenization 

tokenized_doc = df_test['text'].apply(lambda x: x.split()) 
# remove stop-words 

tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 

                                    item not in stop_words]) 
for i in range(len(df_test)):

    t =' '.join(tokenized_doc[i]) 

    detokenized_doc.append(t) 
test_df.head()
test_df['text'] = detokenized_doc

test_df.head()
from sklearn.model_selection import train_test_split 

# split data into training and validation set 

df_trn, df_val = train_test_split(df, stratify = df['label'],  test_size = 0.2, random_state = 12)

df_trn.shape, df_val.shape, test_df.shape
#data_lm = (TextList.from_csv(path, '/kaggle/working/train.csv', cols='text') 

#                   .random_split_by_pct(0.1)

#                   .label_for_lm()

#                   .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='text'))

#                   .databunch())
# Language model data 

data_lm = TextLMDataBunch.from_df(train_df = df_trn, 

                                  valid_df = df_val,

                                  test_df = test_df,

                                  text_cols=['text'],

                                  path = "") 
# Classifier model data 

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, 

                                      valid_df = df_val,

                                      test_df = test_df,

                                      vocab=data_lm.train_ds.vocab, 

                                      bs=32)
learn = language_model_learner(data_lm, pretrained=True,arch=AWD_LSTM,pretrained_model=URLs.WT103, drop_mult=0.7)

#learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
#learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, model_dir="/tmp/model/", drop_mult=0.3)

learn.model
learn.lr_find()

learn.recorder.plot()
#learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))

#learn.fit_one_cycle(1, 1e-1)

#learn.save('mini_train_lm')

#learn.save_encoder('mini_train_encoder')
# train the learner object with learning rate = 1e-2 

learn.fit_one_cycle(3, 1e-2)
#learn.unfreeze()

#learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))
learn.predict("This is a review about", n_words=10)
#learn.show_results()
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, drop_mult=0.7) 

learn.load_encoder('ft_enc')
import torch

from torch import nn

import torch.nn.functional as F



class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduction = reduction



    def forward(self, inputs, targets):

        #print("inputs",inputs.shape)

        #print("target",targets.shape)

        if self.logits:

            BCE_loss = F.cross_entropy_with_logits(inputs, targets, reduction='none')

            #BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        else:

            #BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduction is None:

            return F_loss

        else:

            return torch.mean(F_loss)
learn.loss_func = FocalLoss()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 1e-3)
learn.unfreeze()

learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))
# and plot the losses of the first cycle

learn.recorder.plot_losses()
# get predictions 

preds, targets = learn.get_preds(DatasetType.Valid) 

predictions = np.argmax(preds, axis = 1) 
%matplotlib inline

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=2)

#predictions = model.predict(X_test, batch_size=1000)



LABELS = ['negative','somewhat negative','neutral','somewhat positive','positive'] 



confusion_matrix = metrics.confusion_matrix(targets, predictions)



plt.figure(figsize=(10, 10))

sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});

plt.title("Confusion matrix", fontsize=20)

plt.ylabel('True label', fontsize=20)

plt.xlabel('Predicted label', fontsize=20)

plt.show()
#??TextList.from_csv
#data_clas = (TextList.from_csv(path, '/kaggle/working/train.csv',cols='text', vocab=data_lm.vocab) #test='test'

#    .split_from_df(col='is_valid') #is_valid

#    .label_from_df(cols='target')

#    .add_test(TextList.from_csv(path, '/kaggle/working/test.csv', cols='text'))

#    .databunch(bs=42))
#type(data_clas.test_dl)
#data_clas.show_batch()
#??text_classifier_learner()
#data_clas.c
#len(data_clas.vocab.itos)
#learn = text_classifier_learner(data_clas, drop_mult=0.5) #metrics=[accuracy_thresh]

#learn.load_encoder('mini_train_encoder')

#learn.freeze()

#learn.model
#learn.lr_find()
#learn.recorder.plot()
#learn.crit = F.binary_cross_entropy

#learn.crit = F.binary_cross_entropy_with_logits
#learn.metrics = [accuracy, fbeta] #r2_score, top_k_accuracy
#learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

#learn.fit_one_cycle(1, slice(1e-3,1e-2))

#learn.save('mini_train_clas')
#learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
#learn.freeze_to(-2)

#learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))



#learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
#learn.freeze_to(-3)

#learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
#learn.unfreeze()

#learn.fit_one_cycle(15, slice(2e-3/100, 2e-3))



#learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
# get predictions

#preds, targets = learn.get_preds()



#predictions = np.argmax(preds, axis = 1)

#pd.crosstab(predictions, targets)
#learn.show_results(rows=20)
# Language model data

#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")



# Classifier model data

#data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)
#type(learn.data.test_dl)
probs, _ = learn.get_preds(DatasetType.Test)
probs.shape
probs[0]
preds = probs.argmax(dim=1)
#preds = np.argmax(probs, axis=1)
ids = df_test["PhraseId"].copy()
submission = pd.DataFrame(data={

    "PhraseId": ids,

    "Sentiment": preds

})

submission.to_csv("submission.csv", index=False)

submission.head(n=10)
#df.head()
#from sklearn.model_selection import train_test_split



# split data into training and validation set

#df_trn, df_val = train_test_split(df, stratify = df['target'], test_size = 0.4, random_state = 12)
# Language model data

#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "../input")
# Classifier model data

#data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)