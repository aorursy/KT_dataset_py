# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing libraries and magic functions



import matplotlib.pyplot as plt

import seaborn as sns





from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
# read dataset

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')



# check information and first glimpse at dataframe

df.head()

df.info()

df.columns
# We will rename the target variable class since class is also a built in function in Python/Pandas. 

df = df.rename(columns={"class": "poison"})
# Distribution target variable

df.poison.value_counts()

plt.box(False)

sns.countplot(df['poison'])

plt.title("Distribution of Target Variable", fontweight='bold')
# checking for null values

df.isnull().sum()



# checking for duplicate values

duplicate_df = df[df.duplicated()]

duplicate_df
# how many different values are there within the attributes?

df.nunique()
# closer look at veil-type since there is only 1 veil-type

df['veil-type'].unique()
# dropping veil-type from the dataframe

df = df.drop(['veil-type'],axis=1)
# test-train data split



from sklearn.model_selection import train_test_split



# clarify what is y and what is x label

y = df['poison']

X = df.drop(['poison'], axis = 1)



# divide train test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)



# summarize

print('Train', X_train.shape, y_train.shape)

print('Test', X_test.shape, y_test.shape)
# Creating dummies



from sklearn.preprocessing import OrdinalEncoder

# prepare input data

def prepare_inputs(X_train, X_test):

    oe = OrdinalEncoder()

    oe.fit(X_train)

    X_train_enc = oe.transform(X_train)

    X_test_enc = oe.transform(X_test)

    return X_train_enc, X_test_enc



from sklearn.preprocessing import LabelEncoder

# prepare target

def prepare_targets(y_train, y_test):

    le = LabelEncoder()

    le.fit(y_train)

    y_train_enc = le.transform(y_train)

    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc
# Assigning new dummy variables to train & test data



# prepare input data

X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

# prepare output data

y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
# Feature Selection



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



def select_features(X_train, y_train, X_test):

    fs = SelectKBest(score_func=chi2, k=7)

    fs.fit(X_train, y_train)

    X_train_fs = fs.transform(X_train)

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs



X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)



dfscores = pd.DataFrame(fs.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(7,'Score'))  #print 10 best features





featureScores.plot(kind='bar')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# fit the model

lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train_fs, y_train_enc)

# evaluate the model

yhat = lr.predict(X_test_fs)

# evaluate predictions

accuracy = accuracy_score(y_test_enc, yhat)

print('Accuracy: %.2f' % (accuracy*100))