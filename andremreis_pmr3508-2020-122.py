#Importing necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
#Defining the training data



train_data = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?")

train_data.set_index('Id', inplace=True)
#Defining the test data

test_data = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?")

test_data.set_index('Id', inplace=True)
#Dropping useless columns

train_data.drop(columns=["fnlwgt"], inplace=True)

test_data.drop(columns=["fnlwgt"], inplace=True)
#Checking parameter types

train_data.dtypes
#Printing current dataset

train_data.head()
#Separating parameters by types (object x int64)

obj_p = train_data.select_dtypes(['object']).columns

int_p = train_data.select_dtypes(['int64']).columns



print("object type data")

print(obj_p)

print("---")

print("int64 type data")

print(int_p)
#Data description

train_data.describe()
#Checking what columns have missing data

print(train_data.isnull().sum())
#Taking a deeper look on the columns 'workclass', 'occupation' and 'native.country'

print("workclass")

print(train_data['workclass'].describe())

print("\n---\n")



print("occupation")

print(train_data['occupation'].describe())

print("\n---\n")



print("native.country")

print(train_data['native.country'].describe())
train_data.dropna(subset=["occupation"], inplace=True)

train_data.isnull().sum()



wc_mode = train_data['workclass'].mode()[0]

train_data['workclass'].fillna(wc_mode, inplace=True)



nc_mode = train_data['native.country'].mode()[0]

train_data['native.country'].fillna(nc_mode, inplace=True)
#Checking for remaining missing data - if all values on the right = 0 --> success

train_data.isnull().sum()
train_obj_p = train_data[obj_p].apply(pd.Categorical)



for col in obj_p:

    train_data[col + "_cat"] = train_obj_p[col].cat.codes



##################



test_obj_p = test_data[obj_p[:-1]].apply(pd.Categorical)



for col in obj_p[:-1]:

    test_data[col + "_cat"] = test_obj_p[col].cat.codes



##################

print("train_data")

train_data.dtypes

print("\n---\n")

print("test_data")

test_data.dtypes

sns.heatmap(train_data.loc[:, [*int_p, 'income_cat']].corr().round(2), vmin = -1., vmax = 1., 

            cmap = plt.cm.RdYlGn_r, annot = True)
fig, axes = plt.subplots(nrows = 3, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



train_data.groupby(['sex', 'income']).size().unstack().plot(kind = 'bar', stacked = True, ax = axes[0, 0], figsize = (20, 15))



relationship = train_data.groupby(['relationship', 'income']).size().unstack()

relationship['sum'] = train_data.groupby('relationship').size()

relationship = relationship.sort_values('sum', ascending = False)[['<=50K', '>50K']]

relationship.plot(kind = 'bar', stacked = True, ax = axes[0, 1])



education = train_data.groupby(['education', 'income']).size().unstack()

education['sum'] = train_data.groupby('education').size()

education = education.sort_values('sum', ascending = False)[['<=50K', '>50K']]

education.plot(kind = 'bar', stacked = True, ax = axes[1, 0])



occupation = train_data.groupby(['occupation', 'income']).size().unstack()

occupation['sum'] = train_data.groupby('occupation').size()

occupation = occupation.sort_values('sum', ascending = False)[['<=50K', '>50K']]

occupation.plot(kind = 'bar', stacked = True, ax = axes[1, 1])



workclass = train_data.groupby(['workclass', 'income']).size().unstack()

workclass['sum'] = train_data.groupby('workclass').size()

workclass = workclass.sort_values('sum', ascending = False)[['<=50K', '>50K']]

workclass.plot(kind = 'bar', stacked = True, ax = axes[2, 0])



race = train_data.groupby(['race', 'income']).size().unstack()

race['sum'] = train_data.groupby('race').size()

race = race.sort_values('sum', ascending = False)[['<=50K', '>50K']]

race.plot(kind = 'bar', stacked = True, ax = axes[2, 1])
import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
xColumns = train_data.select_dtypes(include=[np.number]).columns

xColumns = xColumns.drop("income_cat")
x_train = train_data[xColumns]

y_train = train_data.income
%%time



#Getting best K



score_rec = 0.0



for k in range(25, 35):

    knn = KNeighborsClassifier(k, metric = 'manhattan')

    score = np.mean(cross_val_score(knn, x_train, y_train, cv = 10))

    if score > score_rec :

        best_k = k

        score_rec = score



print("Best K: {} | Accuracy: {}".format(best_k, score))
knn = KNeighborsClassifier(best_k, metric = 'manhattan')

knn.fit(x_train, y_train)
%%time



x_test = test_data[xColumns]

y_test = knn.predict(x_test)

y_test
prediction = pd.DataFrame(y_test)

prediction.columns=['income']

prediction['Id'] = prediction.index

prediction = prediction[['Id','income']]

prediction.head()
prediction.to_csv('prediction.csv',index = False)