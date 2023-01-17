!pip install ftfy

!pip install gensim
#Para o uso geral

import random

import numpy as np

import pandas as pd

import copy 

import time

from scipy.stats import uniform

import matplotlib.pyplot as plt

import seaborn as sns

import requests

import io

from subprocess import check_output



#Para o processamento de textos

from ftfy import fix_text

import string

import re

from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument



#Para Machine Learning e NLP

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
path="../input/sentiment-analysis-pmr3508"



print(check_output(["ls", path]).decode("utf8"))
%%time

data = pd.read_csv(path+"/data_train.csv")
data.head()
data.loc[data.duplicated(subset='review', keep=False)==True].sort_values(by='review').head(10)
data=data.drop_duplicates(subset='review', keep='first')

data.shape
data.loc[:,'positive'].value_counts()
X_train = data.loc[:,'review'].tolist()

y_train = np.array(data.loc[:,'positive'].tolist())
X_train[5]
def clean(text):

    txt=text.replace("<br />"," ") #retirando tags

    txt=fix_text(txt) #consertando Mojibakes (Ver https://pypi.org/project/ftfy/)

    txt=txt.lower() #passando tudo para minúsculo

    txt=txt.translate(str.maketrans('', '', string.punctuation)) #retirando toda pontuação

    txt=txt.replace(" — ", " ") #retirando hífens

    txt=re.sub("\d+", ' <number> ', txt) #colocando um token especial para os números

    txt=re.sub(' +', ' ', txt) #deletando espaços extras

    return txt
%%time

X_train = [clean(x) for x in X_train]
X_train = [x.split() for x in X_train]
%%time

d2v = Doc2Vec.load(path+"/doc2vec")  
def emb(txt, model, normalize=False): 

    model.random.seed(42)

    x=model.infer_vector(txt, steps=20)

    

    if normalize: return(x/np.sqrt(x@x))

    else: return(x)
%%time

X_train = [emb(x, d2v) for x in X_train] 

X_train = np.array(X_train)
X_train.shape, y_train.shape
%%time

logreg = LogisticRegression(solver='liblinear',random_state=42)

hyperparams = dict(C=np.linspace(0,10,100), 

                     penalty=['l2', 'l1'])

clf = RandomizedSearchCV(logreg, hyperparams, scoring='roc_auc', n_iter=50, cv=2, n_jobs=-1, random_state=0, verbose=2)

search_logreg = clf.fit(X_train, y_train)
search_logreg.best_params_, search_logreg.best_score_ 
logreg = LogisticRegression(C=search_logreg.best_params_['C'], 

                            penalty=search_logreg.best_params_['penalty'],

                            solver='liblinear', random_state=42)



logreg.fit(X_train, y_train)
%%time

test1 = pd.read_csv(path+"/data_test1.csv")

X_test1 = test1.loc[:,'review'].tolist()

X_test1 = [clean(x).split() for x in X_test1]

X_test1 = [emb(x, d2v) for x in X_test1] 

X_test1 = np.array(X_test1)



y_test1 = np.array(test1.loc[:,'positive'].tolist())



X_test1.shape, y_test1.shape
print('AUCs --- Log. Reg.: {:.4f}'.format(roc_auc_score(y_test1, logreg.predict_proba(X_test1)[:,1])))
%%time

test2 = pd.read_csv(path+"/data_test2_X.csv")

X_test2 = test2.loc[:,'review'].tolist()

X_test2 = [clean(x).split() for x in X_test2]

X_test2 = [emb(x, d2v) for x in X_test2] 

X_test2 = np.array(X_test2)



X_test2.shape
output = {'positive': logreg.predict_proba(X_test2)[:,1]}

output = pd.DataFrame(output)



output.head()
output.to_csv("submission.csv", index = True, index_label = 'Id')