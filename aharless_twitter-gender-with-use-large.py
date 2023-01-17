import os

print(os.listdir("../input"))
TRAIN_INPUT = 'twitgen_train_201906011956.csv'

VALID_INPUT = 'twitgen_valid_201906011956.csv'

TEST_INPUT = 'twitgen_test_201906011956.csv'

EMBEDDING_DIM = 512

MAXLEN = 50
import tensorflow as tf

import tensorflow_hub as hub

import os

import re

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc

from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt

from datetime import datetime
basepath = "/kaggle/input/"
!ls $basepath
df_train = pd.read_csv(basepath+TRAIN_INPUT, index_col=['id','time'], parse_dates=['time'])

df_valid = pd.read_csv(basepath+VALID_INPUT, index_col=['id','time'], parse_dates=['time'])

df_test = pd.read_csv(basepath+TEST_INPUT, index_col=['id','time'], parse_dates=['time'])

df_train.head()
def prepare_data(df):

    text = df['text'].tolist()

    text = [' '.join(t.split()[0:MAXLEN]) for t in text]

    text = np.array(text, dtype=object)

    label = df['male'].tolist()

    return(text, label)

    

train_text, y_train = prepare_data(df_train)    

valid_text, y_valid = prepare_data(df_valid)

test_text, y_test = prepare_data(df_test)
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"



def get_embeddings(text):

    with tf.Graph().as_default():

      embed = hub.Module(module_url)

      embeddings = embed(text)

      with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(tf.tables_initializer())

        return(sess.run(embeddings))



# USEL needs lots of memory, so have to split train file in half

chunk = int(len(train_text)/2)

X_train = np.concatenate([get_embeddings(train_text[:chunk]),

                              get_embeddings(train_text[chunk:])])

X_valid = get_embeddings(valid_text)

X_test = get_embeddings(test_text)

    

X_train.shape, X_valid.shape, X_test.shape
X_train[:5,:5], X_valid[:5,:5], X_test[:5,:5]
pca = PCA(64)

pca.fit(X_train)

print(pca.explained_variance_ratio_)
model = Pipeline([('pca',  PCA(50)),

                  ('poly', PolynomialFeatures()),

                  ('lr',   LogisticRegression(C=.08))])

model = model.fit(X_train, y_train)
y_train_pred = model.predict_proba(X_train)[:,1]

f1_score(y_train, y_train_pred>.5)
y_pred = model.predict_proba(X_valid)[:,1]

print( confusion_matrix(y_valid, (y_pred>.5)) )

f1_score(y_valid, y_pred>.5)
accuracy_score(y_valid, y_pred>.5)
y_test_pred = model.predict_proba(X_test)[:,1]

print( confusion_matrix(y_test, (y_test_pred>.5)) )

f1_score(y_test, y_test_pred>.5)
accuracy_score(y_test, y_test_pred>.5)
fpr, tpr, _ = roc_curve(y_test, y_test_pred)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='darkorange',

         lw=1, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')

plt.legend(loc="lower right")

plt.show()
pd.Series(y_test_pred).hist()

plt.show()
df_acc = pd.DataFrame(columns=['minprob','maxprob','count','accuracy'])

for pbot in np.linspace(0,.9,10):

    ptop = pbot+.1

    mask = (y_test_pred>=pbot)&(y_test_pred<ptop)

    count = int(mask.sum())

    if count>0:

        actual = pd.Series(y_test)[mask].values

        pred_prob = pd.Series(y_test_pred)[mask].values

        pred_bin = pred_prob>.5

        acc = accuracy_score(actual, pred_bin)

        nsucc = sum(actual==pred_bin)

        confint = proportion_confint(nsucc, count)

        minconf = confint[0]

        maxconf = confint[1]

    else:

        acc = np.nan

        minconf = np.nan

        maxconf = np.nan

    row = pd.DataFrame({'minprob':[pbot], 'maxprob':[ptop], 'count':[count], 

                        'accuracy':[acc], 'lconf95':[minconf], 'hconf95':[maxconf]})

    df_acc = pd.concat([df_acc, row], sort=False)

df_acc.set_index(['minprob','maxprob'])
df_acc['avgprob'] = .5*(df_acc.minprob+df_acc.maxprob)

ax = df_acc.drop(['count','minprob','maxprob'],axis=1).set_index('avgprob').plot(

        title='Accuracy of Predictions by Range')

ax.legend(labels=['accuracy', '95% conf, lower', '95% conf, upper'])

ax.set(xlabel="center of probability bin", ylabel="fraction of correct predicitons")

plt.show()