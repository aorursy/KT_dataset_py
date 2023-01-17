import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import more_itertools as mit

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing as pre

from sklearn.preprocessing import StandardScaler as ssc

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder





from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

import time
#Load the data which was changed from .data to .csv

print(os.listdir("../input"))

datr = pd.read_csv("../input/K8.csv",low_memory=False)
datr.describe()
datr.head()
#Remove the column title of datr, make it row one, then make the column title integer values

datrcolumn=pd.DataFrame(list(datr.columns)).T

datr.columns = datrcolumn.columns

datrefined=datrcolumn.append(datr, ignore_index = True)



#Remove the last column of datrefined

datrefined.drop(columns = list(datrefined.columns)[-1],inplace=True)
#Remove the second decimal points with the digits that follows it

for i in range(0,len(datrefined.iloc[0])):

    datrefined[i]=datrefined[i].apply(lambda x: x[:list(mit.locate(x, lambda s: s == "."))[1]] if x.count('.') > 1 else x)
# Change string numerical values to numbers

datrefined.iloc[:,:-1]=datrefined.iloc[:,:-1].apply(pd.to_numeric, errors='coerce')

# Drop all rows NaN

datrefined.dropna(axis=0, how='all', inplace=True)

# Drop all columns NaN

datrefined.dropna(axis=1, how='all', inplace=True)

# Drop rows/ with over 90% of the column with NaN

datrefined.dropna(axis=0, thresh=datrefined.shape[1]*90//100, inplace=True)

# Print the total number of rows dropped.

print('Total number of rows dropped is '+str(datr.shape[0]-datrefined.shape[0]))

# Check if there are still missing missing data in the dataset

missingSum = datrefined.isnull().sum().sum()

print('The total number of missing values is: ',missingSum)
datrefined.index = list(range(0,datrefined.shape[0]))
#Encoding categorical data values

labelencoder_y = LabelEncoder()

datrefined[datrefined.columns[-1]] = labelencoder_y.fit_transform(datrefined[datrefined.columns[-1]])
datrefined.rename(columns={datrefined.columns[-1]:'target'}, inplace=True)

y = datrefined['target'].values

X = datrefined.drop('target', axis=1).values



X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.20, random_state=21)
# Standardizing the features

X = ssc().fit_transform(X)

X_train = ssc().fit_transform(X_train)

X_test = ssc().fit_transform(X_test)
pcaPlot = PCA(n_components=2)

pcaPlot.fit(X)

principalComponents = pcaPlot.transform(X)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
pcaForPloting = pd.concat([principalDf, datrefined['target']], axis = 1, join='inner')

pcaForPloting.rename(columns={datrefined.columns[-1]:'target'}, inplace=True)
pcaForPloting.head()
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targetss = ['inactive', 'active']

targets = [1, 0]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = pcaForPloting['target'] == target

    ax.scatter(pcaForPloting.loc[indicesToKeep, 'principal component 1']

               , pcaForPloting.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targetss)

ax.grid()
#Fitting the PCA algorithm with our Data

pcaModel = PCA().fit(X)



#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pcaModel.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Dataset Explained Variance for Training Data')

plt.show()



#Reducing the data dimension to retain .90 variance

varianceToPreserve = 0.90



pca = PCA(varianceToPreserve)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)



print('Selecting '+str(X_train.shape[1])+' X_train components we can preserve '+str(varianceToPreserve)+' of the total variance of the data.')

print('Selecting '+str(X_test.shape[1])+' X_test components we can preserve '+str(varianceToPreserve)+' of the total variance of the data.')
# Building a list of models to use

models_list = []

models_list.append(('CART', DecisionTreeClassifier()))

models_list.append(('SVM', SVC())) 

models_list.append(('NB', GaussianNB()))

models_list.append(('KNN', KNeighborsClassifier()))
# Building the Models

num_folds = 4

results = []

names = []



for name, model in models_list:

    kfold = KFold(n_splits=num_folds, random_state=123)

    start = time.time()

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    end = time.time()

    results.append(cv_results)

    names.append(name)

    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
# Making Performance plots

fig = plt.figure()

fig.suptitle('Performance Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# prepare the model

predictions = []

for name, model in models_list:

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    print( "Run Time for %s is : %f" % (name, end-start))

    

    # estimate accuracy on test dataset

    pred = model.predict(X_test)

    predictions.append((name, pred))

for name, pred in predictions:

    print('Results for '+name+' :::')

    print("  Accuracy score %f" % accuracy_score(y_test, pred))

    print(classification_report(y_test, pred))

    cm = np.array(confusion_matrix(y_test, pred, labels=[1,0]))

    confusion = pd.DataFrame(cm, index=['Is Inactive','Is Active'],

                             columns=['Predicted Inactive','Predicted Active'])

    #sns.heatmap(confusion, annot=True)

    #confusion.plot_confusion_matrix(title='Confusion matrix')

    print(confusion)

    print('     ')

    print('     ')

    print('     ')