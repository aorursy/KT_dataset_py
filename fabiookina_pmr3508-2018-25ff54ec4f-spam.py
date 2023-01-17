# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sklearn 
import os
print(os.listdir("../input/data-spam"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/data-spam/train_data.csv")
test_data = pd.read_csv("../input/data-spam/test_features.csv")
train_data
test_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
X_train_data = train_data.drop(['ham','Id'],axis=1)
Y_train_data= train_data.loc[:,'ham']
X_train_data
Y_train_data
acertos=[]
for i in range (1,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    acerto=cross_val_score(knn, X_train_data, Y_train_data, cv=20)
    acertos.append(acerto.mean())
acertos
max(acertos)
k=acertos.index(max(acertos))+1
k
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train_data, Y_train_data)
Y_Pred = knn.predict(test_data.drop('Id',axis=1))

results_knn = pd.DataFrame()
results_knn['Id'] = test_data['Id']
results_knn['ham'] = Y_Pred
results_knn
results_knn.to_csv("results_knn.csv")
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train_data, Y_train_data)
cross_val_score(GNB, X_train_data, Y_train_data, cv=20).mean()

Y_Pred_gauss = GNB.predict(test_data.drop('Id',axis=1))
results_GNB = pd.DataFrame()
results_GNB['Id'] = test_data['Id']
results_GNB['ham'] = Y_Pred_gauss
results_GNB

results_GNB.to_csv("results_GNB.csv")
#using multinomial NB for word count type of data, excluding data from capital letter sequences count
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train_data.drop(['capital_run_length_average','capital_run_length_longest','capital_run_length_total'],axis=1), Y_train_data)
cross_val_score(MNB, X_train_data.drop(['capital_run_length_average','capital_run_length_longest','capital_run_length_total'],axis=1), Y_train_data, cv=20).mean()
Y_Pred_multinomial = MNB.predict(test_data.drop(['capital_run_length_average','capital_run_length_longest','capital_run_length_total','Id'],axis=1))
results_MNB = pd.DataFrame()
results_MNB['Id'] = test_data['Id']
results_MNB['ham'] = Y_Pred_multinomial
results_MNB

results_MNB.to_csv("results_MNB.csv")