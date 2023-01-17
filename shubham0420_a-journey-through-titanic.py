import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

train.head()
train.info()

print("----------------------------")

test.info()
train['Sex'] = train['Sex'].astype('str')
train.head()
train = train.drop(['PassengerId'],axis = 1)
name = train['Name'].values

a = []

for i in name:

    a.append(i.split(',')[1])

b = []

for i in a:

    b.append(i.split('.')[0])

for i in range(len(b)):

    b[i] = b[i].strip()

dictionary = {'Name':b}
train['Name'] = dictionary['Name']
'''

for i in train['Name'][:]:

    if str(i) == str('Mlle'):

        train['Name'][i] = 'Miss'

    if str(i) == str('Ms'):

        train['Name'][i]= 'Miss'

    if str(i) == str('Mme'):

        train['Name'][i] ='Mrs'

'''
rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

train.loc[train["Name"] == "Mlle", "Name"] = str('Miss')

train.loc[train["Name"] == "Ms", "Name"] = str('Miss')

train.loc[train["Name"] == "Mme", "Name"] = str('Mrs')

for i in rare_title:

    train.loc[train['Name']==i,'Name'] = str('Rare_title')
for i in train['Name']:

    train['Name'][i]=str(i)
train.head()
