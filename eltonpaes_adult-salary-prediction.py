# Importing libraries to play with data

import numpy as np

import pandas as pd
# Importing the data itself

data_set  = pd.read_csv('../input/adult.csv')
data_set.info()
data_set.head()
# Let's see how many unique categories we have in this property

occupation_set = set(data_set['occupation'])

print(occupation_set)
# Now we classify them as numers instead of their names.

data_set['occupation'] = data_set['occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2, 

                                                       'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,

                                                       'Machine-op-inspct': 6, 'Exec-managerial': 7, 

                                                       'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10, 

                                                       'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13, 

                                                       'Protective-serv': 14}).astype(int)
# Just print it to see if nothing gone wrong

data_set.head()
# Again, let's see how many unique categories we have in this property

income_set = set(data_set['income'])

print(income_set)
# As expected. Just transforming now.

data_set['income'] = data_set['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
# Just print it to see if nothing gone wrong

data_set.head()
# Importing matlab to plot graphs

import matplotlib as plt

%matplotlib inline
data_set.groupby('education').income.mean().plot(kind='bar')
data_set.groupby('occupation').income.mean().plot(kind='bar')
from sklearn.model_selection import train_test_split



# Taking only the features that is important for now

X = data_set[['education.num', 'occupation']]



# Taking the labels (Income)

Y = data_set['income']



# Spliting into 80% for training set and 20% for testing set so we can see our accuracy

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Importing C-Support Vector Classification from scikit-learn

from sklearn.svm import SVC



# Declaring the SVC with no tunning

classifier = SVC()



# Fitting the data. This is where the SVM will learn

classifier.fit(X_train, Y_train)



# Predicting the result and giving the accuracy

score = classifier.score(x_test, y_test)



print(score)
# Transforming the Sex into 0 and 1

data_set['sex'] = data_set['sex'].map({'Male': 0, 'Female': 1}).astype(int)
# How many unique races we got here?

race_set = set(data_set['race'])

print(race_set)
data_set['race'] = data_set['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 

                                             'Amer-Indian-Eskimo': 4}).astype(int)
# What about maritial status?

mstatus_set = set(data_set['marital.status'])

print(mstatus_set)
data_set['marital.status'] = data_set['marital.status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 

                                                             'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4, 

                                                             'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
# Everythin' good?

data_set.head()
import seaborn as sns

import matplotlib.pyplot as pplt

#correlation matrix

corrmat = data_set.corr()

f, ax = pplt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 8 #number of variables for heatmap

cols = corrmat.nlargest(k, 'income')['income'].index

cm = np.corrcoef(data_set[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

pplt.show()
# Taking only the features that is important for now

X = data_set[['education.num', 'age']]



# Taking the labels (Income)

Y = data_set['income']



# Spliting into 80% for training set and 20% for testing set so we can see our accuracy

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Declaring the SVC with no tunning

classifier = SVC()



# Fitting the data. This is where the SVM will learn

classifier.fit(X_train, Y_train)



# Predicting the result and giving the accuracy

score = classifier.score(x_test, y_test)



print(score)
# Taking only the features that is important for now

X = data_set[['education.num', 'age', 'hours.per.week', 'capital.gain']]



# Taking the labels (Income)

Y = data_set['income']



# Spliting into 80% for training set and 20% for testing set so we can see our accuracy

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Declaring the SVC with no tunning

classifier = SVC()



# Fitting the data. This is where the SVM will learn

classifier.fit(X_train, Y_train)



# Predicting the result and giving the accuracy

score = classifier.score(x_test, y_test)



print(score)
data_set.groupby('race').income.mean().plot(kind='bar')
data_set.groupby('sex').income.mean().plot(kind='bar')
# Mean below 20 years old

data_set.groupby('age').income.mean().plot(kind='bar')