## Lib imports



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
## File locations and names

file_location = '/kaggle/input/adult-pmr3508/'

train_filename = "train_data.csv"

test_filename = "test_data.csv"
## Opening dataset file

adult = pd.read_csv(file_location + train_filename, na_values = "?", header=0, index_col = 0)

adult.head()
## Lists missing values by collumns

adult.isnull().sum()
adult["native.country"].value_counts()

adult["workclass"].value_counts()

adult["occupation"].value_counts()

## Sex comparison 

plt.figure(figsize=(12,12))

plt.suptitle("Sex comparison betwen income ranges")

plt.subplot(221)

adult[adult['income'] == '<=50K'].sex.value_counts().plot(kind = "pie")

plt.title("income <=50K")

plt.subplot(222)

adult[adult['income'] == '>50K'].sex.value_counts().plot(kind = "pie")

plt.title("income >50K")

plt.show()
def splitByColl(ds,coll, reduced_ticks = 0):

    #Plots stack bar graphs for categorical collumns

    

    vars = ds[coll].drop_duplicates().values

    split_colls = []

    split_gt50 = []

    split_leq50 = []

    

    for var in vars:

        split_colls.append(var)

        aux = ds[ds[coll] == var]        

        split_leq50.append(len(aux[aux['income'] == '<=50K']))

        split_gt50.append(len(aux[aux['income'] == '>50K']))

    

    plt.figure(figsize=(10,10))

    ind = np.arange(len(split_colls))

    width = 0.5

    if reduced_ticks:

        for i in range( len(split_colls)):

            if i%reduced_ticks != 0:

                split_colls[i] = ' '



    b1 = plt.bar(ind, split_leq50)

    b2 = plt.bar(ind, split_gt50)

    plt.title("income by "+ coll)

    plt.legend((b1[0], b2[0]), ('<=50K', '>50K'))

    plt.xticks(ind,split_colls, rotation = 90)

    plt.show()



splitByColl(adult.dropna(), 'workclass')

splitByColl(adult.dropna(), 'occupation')

splitByColl(adult, 'education')

splitByColl(adult.sort_values(by=['age']), 'age', 10)

splitByColl(adult, 'race')
## Assign numerical values to categorical collumns - Useful for data analysis

nAdult = adult.dropna().apply(sklearn.preprocessing.LabelEncoder().fit_transform)

nAdult.describe()
## Correlation map

correlation = nAdult.corr()

plt.figure(figsize=(15,15))

sns.heatmap(correlation, annot=True, cmap= 'Pastel1')
## Opening test ds file

test_adult = pd.read_csv(file_location + test_filename, na_values = "?", header=0, sep=r'\s*,\s*', engine = 'python', index_col = 0)

test_adult.head()
## Replaces missing values with collumn mode



for coll in adult:

    adult[coll] = adult[coll].fillna(adult[coll].mode())

    

X_data_numeric = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Y_data_numeric = adult.income
## knn instancing and cross validation

knn_numeric = KNeighborsClassifier(n_neighbors=16)

score_numeric = cross_val_score(knn_numeric, X_data_numeric, Y_data_numeric, cv=10)

print(score_numeric)
## Missing values replacement

for coll in test_adult:

    test_adult[coll] = test_adult[coll].fillna(test_adult[coll].mode())

    

X_test_numeric = test_adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
knn_numeric.fit(X_data_numeric, Y_data_numeric)
## Prediction

Y_prediction_numeric = knn_numeric.predict(X_test_numeric)

print(Y_prediction_numeric)
## Generates submission file

prediction = pd.DataFrame()



prediction[0] = test_adult.index

prediction[1] = Y_prediction_numeric

prediction.columns = ['Id','Income']

print(prediction)

prediction.to_csv('prediction_k16.csv',index = False)

## Attempt to use relationship for the classifier by labeling its values



relationship = adult.relationship

le = preprocessing.LabelEncoder()

le.fit(relationship)

list(le.classes_)

nrelationship = le.transform(relationship)

mod_adult = adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 'income']]

mod_adult.insert(5,'relationship', nrelationship, True)
X_mod_adult = mod_adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week", 'relationship']]

X_mod_adult.shape

Y_mod_adult = mod_adult.income

Y_mod_adult.shape
## Instancing of knn for relationship added model

knn_rel = KNeighborsClassifier(n_neighbors=15, n_jobs = -1)

knn_rel.fit(X_mod_adult, Y_mod_adult)

cross_val_score(knn_rel, X_mod_adult, Y_mod_adult, cv=10, n_jobs = -1)
## Labeling of test data and prediction for it



X_test_mod = test_adult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

ntestrelationship = le.transform(test_adult.relationship)

X_test_mod.insert(5,'relationship', ntestrelationship, True)



Y_prediction_mod = knn_rel.predict(X_test_mod)

## Generates submission file

prediction = pd.DataFrame()



prediction[0] = test_adult.index

prediction[1] = Y_prediction_mod

prediction.columns = ['Id','Income']

print(prediction)

prediction.to_csv('prediction_mod_k15.csv',index = False)
