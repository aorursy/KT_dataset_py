# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import csv

import re

from tabulate import tabulate

from collections import defaultdict



# visualization

import seaborn as sns

import matplotlib.pyplot as plt





# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm





# NLP

from textblob import TextBlob







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))




test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")



combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
# preview the data

train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

train_df.describe(include=['O'])
train_df['keyword'].unique()
train_df['location'].unique()

train_df[['keyword', 'target']].groupby(['keyword'], as_index=False).mean()



uniqueKeys=[]

keywordTarget=train_df[['keyword', 'target']].groupby(['keyword'], as_index=False).mean()

for i in range (0,221):

    uniqueKeys.append(keywordTarget['keyword'][i])

print(uniqueKeys)

print("-"*40)

print(len(uniqueKeys))


test_df['keyword'].unique()



train_df[['location', 'target']].groupby(['location'], as_index=False).mean()

print(uniqueKeys)

type(uniqueKeys)
idxlist = []

for i in range (1,223):

    idxlist.append(i)

    

print(idxlist)
mapare = dict(zip(uniqueKeys, idxlist))

print(mapare)
for dataset in combine:

    dataset['keyword'] = dataset['keyword'].map(mapare)

    dataset['keyword'] = dataset['keyword'].fillna(0)
train_df.head()
train_df[['keyword', 'target']].groupby(['keyword'], as_index=False).mean()
cuvinte_text = []

for i in range(0,7613):

    cuvinte_text.append(train_df['text'][i].split())



contor_cuvinte = defaultdict(int)



for doc in cuvinte_text:

    for word in doc:

        contor_cuvinte[word] += 1



PRIMELE_N_CUVINTE = 1000

        

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]

perechi_cuvinte_frecventa = list(contor_cuvinte.items())



# sortam descrescator lista de tupluri dupa frecventa

perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)



# extragem primele 1000 cele mai frecvente cuvinte din toate textele

perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]



print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])

        
list_of_selected_words = []

for cuvant, frecventa in perechi_cuvinte_frecventa:

    list_of_selected_words.append(cuvant)

### numaram cuvintele din toate documentele ###
def get_bow(text, lista_de_cuvinte):

    '''

    returneaza BoW corespunzator unui text impartit in cuvinte

    in functie de lista de cuvinte selectate

    '''

    contor = dict()

    cuvinte = set(lista_de_cuvinte)

    for cuvant in cuvinte:

        contor[cuvant] = 0

    for cuvant in text:

        if cuvant in cuvinte:

            contor[cuvant] += 1

    return contor
def get_bow_pe_corpus(corpus, lista):

    '''

    returneaza BoW normalizat

    corespunzator pentru un intreg set de texte

    sub forma de matrice np.array

    '''

    bow = np.zeros((len(corpus), len(lista)))

    for idx, doc in enumerate(corpus):

        bow_dict = get_bow(doc, lista)

        ''' 

            bow e dictionar.

            bow.values() e un obiect de tipul dict_values 

            care contine valorile dictionarului

            trebuie convertit in lista apoi in numpy.array

        '''

        v = np.array(list(bow_dict.values()))



        bow[idx] = v

    return bow
data_bow = get_bow_pe_corpus(cuvinte_text, list_of_selected_words)

print ("Data bow are shape: ", data_bow.shape)
indici_train = np.array([],dtype=int)

for i in range(0,7613):

    indici_train = np.append(indici_train,i)

    
idx_tr = np.array([],dtype=int)

idx_ts = np.array([],dtype=int)

for i in range (0,7001):

    idx_tr = np.append(idx_tr,indici_train[i])



for i in range (7001,7613):

    idx_ts = np.append(idx_ts,indici_train[i])

labels = np.array([],dtype=int)

for i in range(0,7613):

    labels = np.append(labels,train_df['target'][i])
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = random_forest.predict(data_bow[idx_ts, :])

random_forest.score(data_bow[idx_tr, :], labels[idx_tr])

acc_random_forest = round(random_forest.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_random_forest
# KNN



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = knn.predict(data_bow[idx_ts, :])

acc_knn = round(knn.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_knn
# Gaussian



gaussian = GaussianNB()

gaussian.fit(data_bow, labels)

Y_pred = gaussian.predict(data_bow[idx_ts, :])

acc_gaussian = round(gaussian.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = perceptron.predict(data_bow[idx_ts, :])

acc_perceptron = round(perceptron.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = linear_svc.predict(data_bow[idx_ts, :])

acc_linear_svc = round(linear_svc.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = sgd.predict(data_bow[idx_ts, :])

acc_sgd = round(sgd.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = decision_tree.predict(data_bow[idx_ts, :])

acc_decision_tree = round(decision_tree.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_decision_tree
# Support Vector Machines



svc = SVC()

svc.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = svc.predict(data_bow[idx_ts, :])

acc_svc = round(svc.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_svc
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(data_bow[idx_tr, :], labels[idx_tr])

Y_pred = logreg.predict(data_bow[idx_ts, :])

acc_log = round(logreg.score(data_bow[idx_tr, :], labels[idx_tr]) * 100, 2)

acc_log
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
test_df.head()

test_df.tail()
cuvinte_text_pred = []

for i in range(0,3263):

    cuvinte_text_pred.append(test_df['text'][i].split())
pred_data_bow = get_bow_pe_corpus(cuvinte_text_pred, list_of_selected_words)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(data_bow, labels)

Y_pred = random_forest.predict(pred_data_bow)

print(Y_pred)
submission = pd.DataFrame({

        "id": test_df["id"],

        "target": Y_pred

    })
submission.to_csv("submission.csv",index=False)