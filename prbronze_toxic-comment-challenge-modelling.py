# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import   roc_auc_score,multilabel_confusion_matrix

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier



import warnings

warnings.filterwarnings('ignore')



import nltk

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



directory = '../output'

if not os.path.exists(directory):

    os.makedirs(directory)
os.listdir('/kaggle/output/')
Xtrain = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

Xtest = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')



Xtr = Xtrain[['comment_text']]

ytr = Xtrain[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate']]



Xts = Xtest[['comment_text']]



xtrain, xtest, ytrain, ytest = train_test_split(

    Xtr,ytr,test_size=0.20)

print(xtrain.shape)
#checking distributions

print(ytrain.apply(pd.value_counts))

print(ytest.apply(pd.value_counts))
# Word Vectorizer

vect = TfidfVectorizer(lowercase = True,ngram_range = (1,1),

                       use_idf = True,sublinear_tf = True,

                       stop_words='english',max_features=10000)



clf = OneVsRestClassifier(LogisticRegression(multi_class="multinomial"))



modelo = Pipeline([('vetorizador',vect),

                   ('classificador',clf)

                  ])



modelo.fit(xtrain.comment_text,ytrain);



ypred = np.array(modelo.predict_proba(xtest.comment_text))



roc_auc_score(ytest,ypred,average='macro')



y_pred_sub = np.array(modelo.predict_proba(Xts.comment_text))

y_pred_sub.shape



#checking confusion matrix per label

ypred_class = np.array(

    modelo.predict(xtest.comment_text)

)

multilabel_confusion_matrix(ytest,ypred_class)



#submit ... y_pred_sub using sample_submission.csv