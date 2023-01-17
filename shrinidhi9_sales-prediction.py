# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas_profiling as pp



dataset = pd.read_csv('../input/SalesKaggle3.csv')

dataset['SoldFlag'] = dataset['SoldFlag'].astype('bool')

dataset['New_Release_Flag'] = dataset['New_Release_Flag'].astype('bool')
dataset.drop(['Order', 'MarketingType','SoldCount'], inplace=True, axis=1) 

dataset_train = dataset[dataset['File_Type'] == 'Historical']

dataset_train.drop('File_Type', inplace=True, axis=1) 

dataset_test = dataset[dataset['File_Type'] == 'Active']

dataset_test.drop('File_Type', inplace=True, axis=1) 



# pp.ProfileReport(dataset_train)
dataset_train.info()
print(dataset.isnull().sum())
# pip install phik

import phik

from phik import resources, report

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)



corr = dataset_train.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
print('Correlation with dependent variable')

corr = dataset_train.phik_matrix()['SoldFlag'].abs()

to_drop_1 = [col for col in corr.index if corr[col]<0.05]

dataset_train.drop(to_drop_1, axis=1, inplace=True)

corr = dataset_train.phik_matrix()

col = corr.index

print('Correlation between independent variables')

for i in range(len(col)):

    for j in range(i+1, len(col)):

        if corr.iloc[i,j] >= 0.8:

            print(f"{col[i]} -{col[j]}")
# dataset_train
# pp.ProfileReport(dataset_train)
corr = dataset_train.phik_matrix()

plt.figure(figsize=(10,8))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(corr, annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
#Remove outliers - 

from scipy import stats

numeric_cols = dataset_train.select_dtypes(include=['int64','float64'])

z = np.abs(stats.zscore(numeric_cols))

to_drop_rows=[]



for i in range(numeric_cols.shape[0]):

    for j in range(numeric_cols.shape[1]):

        if z[i,j] >= 3:

            print(f"{i} -{j}")

            to_drop_rows.append(i)

            numeric_cols.iloc[i,j] = numeric_cols.iloc[:,j].median()



# drop or replace by mean

#dataset = dataset.drop([to_drop_rows], axis=0)

dataset_train.update(numeric_cols)    
#For Categorical vars - remove/replace low freq vars



for col in dataset_train.select_dtypes(include=['category','object']).columns:

    dataset_train.loc[dataset_train[col].value_counts()[dataset_train[col]].values < 10, col] = np.nan



dataset_train = pd.get_dummies(dataset_train)



X = dataset_train.drop('SoldFlag',axis=1).values

y = dataset_train['SoldFlag'].values





from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y,random_state = 7,test_size=0.2)



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import model_selection
# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=7)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)