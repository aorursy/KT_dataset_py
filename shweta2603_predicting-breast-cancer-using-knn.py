# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt#data visualization library

from sklearn.feature_selection import SelectKBest #feature selection algorithm

from sklearn.feature_selection import chi2#Compute chi-squared stats between each non-negative feature and class.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filePath = (os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(filePath)

print(data.head())

X = data.iloc[:,3:31]  #independent columns

y = data['diagnosis']   #target column i.e diagnosis 

#apply SelectKBest class to extract top 10 best features

featuresList = []

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

features = featureScores.nlargest(10,'Score')

print(features)  #print 10 best features

for features in features['Specs']:

    featuresList.append(features)#to avoid hardcode later



#Creating new data frame for the best features

data_drop = data[featuresList]
#plot these features against diagnosis to better understand the data and importance of features

for index,columns in enumerate(data_drop):

    plt.figure(index)

    plt.figure(figsize=(15,15))

    sns.stripplot(x='diagnosis', y= columns, data= data, jitter=True, palette = 'Set1').set_title('Diagnosis vs ' + str(columns))

    plt.show()
for index,columns in enumerate(data_drop):

    plt.figure(index)

    plt.figure(figsize=(10,10))

    sns.stripplot(x='diagnosis', y= columns, data= data, jitter=True, palette = 'Set1')

    #sns.plt.title('Diagnosis vs ' + str(columns))

    plt.show()
#Data and label

X = data_drop[:]

y = data['diagnosis']
#implementing KNN

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#scale the features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#train and predict

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)

#predict

y_pred = classifier.predict(X_test)



#evaluate the algorithm

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test,y_pred))
