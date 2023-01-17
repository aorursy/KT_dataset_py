import os,re,zipfile

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import itertools

plt.style.use('seaborn')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train[:3]
train.target.value_counts().plot.bar(rot=0);
import tensorflow as tf

import tensorflow_hub as hub

class UniversalSenteneceEncoder:



    def __init__(self, encoder='universal-sentence-encoder', version='4'):

        self.version = version

        self.encoder = encoder

        self.embd = hub.load(f"https://tfhub.dev/google/{encoder}/{version}",)



    def embed(self, sentences):

        return self.embd(sentences)



    def squized(self, sentences):

        return np.array(self.embd(tf.squeeze(tf.cast(sentences, tf.string))))
%%time

use = UniversalSenteneceEncoder(encoder='universal-sentence-encoder-large',version='5')
%%time

train = train[['text','target']].copy()

train['vects'] = use.squized(train.text).tolist()
%%time

test['vects'] = use.squized(test.text).tolist()
from sklearn import metrics

from sklearn.model_selection import train_test_split

import xgboost as xgb
train_,test_ = train_test_split(train,test_size=0.33,random_state=42,stratify=train.target)
%%time

xgc = xgb.XGBClassifier(n_estimators=300)

xgc.fit(pd.DataFrame(train_.vects.values.tolist()),train_['target'])
%%time

xgc_results = test_.copy()

xgc_results['y_pred'] = xgc.predict(pd.DataFrame(test_.vects.values.tolist()))
print(metrics.classification_report(xgc_results.target,xgc_results.y_pred))
sns.heatmap(metrics.confusion_matrix(xgc_results.target,xgc_results.y_pred),annot=True,fmt='d');
from sklearn.model_selection import GridSearchCV
%%time

xgc_full = xgb.XGBClassifier(n_estimators=300)

xgc_full.fit(pd.DataFrame(train.vects.values.tolist()),train['target'])
%%time

xgc_results_full = test.copy()

xgc_results_full['y_pred'] = xgc_full.predict(pd.DataFrame(test.vects.values.tolist()))
xgc_results_full
xgc_results_full[['id','y_pred']].rename(columns={'y_pred':'target'}).to_csv('submissionv.csv',index=False)