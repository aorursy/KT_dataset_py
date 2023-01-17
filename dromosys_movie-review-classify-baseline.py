# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# import libraries 

import fastai 

from fastai import * 

from fastai.text import * 

import pandas as pd 

import numpy as np 

from functools import partial 

import io 

import os

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')#.sample(frac=0.01)

df_train.head()
df_test = pd.read_csv('../input/test.csv')#.sample(frac=0.01)

df_test.head()
df_train['text'] = df_train['text'].str.replace("[^a-zA-Z]", " ")
df_test['text'] = df_test['text'].str.replace("[^a-zA-Z]", " ")
df_train.target.value_counts()
from sklearn.model_selection import train_test_split 

# split data into training and validation set 

df_trn, df_val = train_test_split(df_train,  test_size = 0.3, random_state = 12) #stratify = df_train.target

df_trn.shape, df_val.shape, df_test.shape
# Language model data 

data_lm = TextLMDataBunch.from_df(train_df = df_trn, 

                                  valid_df = df_val,

                                  test_df = df_test,

                                  text_cols=['text'],

                                  label_cols=['target'],

                                  path = "") 
# Classifier model data 

data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, 

                                      valid_df = df_val,

                                      test_df = df_test,

                                      vocab=data_lm.train_ds.vocab, 

                                      text_cols=['text'],

                                      label_cols=['target'],

                                      bs=64)
learn = language_model_learner(data_lm, pretrained=True,arch=AWD_LSTM, drop_mult=0.7)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.predict("This is a review about", n_words=10)
learn.save_encoder('ft_enc')
learn = text_classifier_learner(data_clas, drop_mult=0.7, arch=AWD_LSTM) 

learn.load_encoder('ft_enc')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, 1e-3)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr = 5e-3

lr
learn.fit_one_cycle(5, slice(lr/100, lr))
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



LABELS = ['1-very negative','2-somewhat negative','3-negative','4-neutral','7-neutral','8-positive','9-somewhat positive','10-very positive'] 



confusion_matrix = metrics.confusion_matrix(targets, predictions)



plt.figure(figsize=(10, 10))

sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});

plt.title("Confusion matrix", fontsize=17)

plt.ylabel('True label', fontsize=17)

plt.xlabel('Predicted label', fontsize=17)

plt.show()
probs, _ = learn.get_preds(DatasetType.Test, ordered=True)
preds = probs.argmax(dim=1)
ids = df_test["id"].copy()
submission = pd.DataFrame(data={

    "id": ids,

    "rating": preds

})

submission.to_csv("submission.csv", index=False)

submission.head(n=10)
submission.rating.value_counts()