# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load data from csv

alldata = pd.read_csv("../input/creditcard.csv")

print (alldata.head())

#Separate data labels

labels = alldata['Class'].as_matrix()



#Clear Time and Class elements, Class because this is the labels,

#Time because we are will make the (controversial) assumption that

#fraudualent and legitamate transactions occur at all hours

alldata.drop('Time', axis = 1, inplace = True)

alldata.drop('Class',axis = 1, inplace = True)



#Make numpy matrix with 29 features

dataset = alldata.iloc[:].as_matrix()

print(dataset.shape)

print(labels.shape)
#See if PCA is an appropriate regularization method

import matplotlib.pyplot as plt

import sklearn.preprocessing as prepro

from sklearn.decomposition import PCA



pca = PCA()

X_pca = pca.fit_transform(prepro.scale(dataset))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title('Scree Plot')

plt.show()
#Naive Bayes Predictive 

from sklearn.naive_bayes import GaussianNB

import sklearn.model_selection as ms



#Cross Val for accuracy

x_train, x_test, y_train, y_test = ms.train_test_split(dataset, labels)



#Naive Bayes

nb_model = GaussianNB()

nb_model.fit(x_train, y_train)

nb_predict = nb_model.predict_proba(x_test)



#ROC Curve

import sklearn.metrics as met

fpr1, tpr1, treshholds = met.roc_curve(y_test,nb_predict[:,1])

x1 = [0,1]

y1 = [0,1]

plt.plot(fpr1,tpr1)

plt.plot(x1,y1)

plt.title('ROC Curve - All Data')

plt.show()

print("AUC: ", met.auc(fpr1,tpr1))

print("Accuracy: ", nb_model.score(x_test,y_test))



#PCA on 5 features

pca = PCA()

X_pca = pca.fit_transform(prepro.scale(dataset))



#Cross Val for accuracy

x_train, x_test, y_train, y_test = ms.train_test_split(X_pca, labels)



#Naive Bayes

nb_model = GaussianNB()

nb_model.fit(x_train, y_train)

nb_predict = nb_model.predict_proba(x_test)



#ROC Curve

import sklearn.metrics as met

fpr1, tpr1, treshholds = met.roc_curve(y_test,nb_predict[:,1])

x1 = [0,1]

y1 = [0,1]

plt.plot(fpr1,tpr1)

plt.plot(x1,y1)

plt.title('ROC Curve - 25 Features')

plt.show()

print("AUC: ", met.auc(fpr1,tpr1))

print("Accuracy: ", nb_model.score(x_test,y_test))