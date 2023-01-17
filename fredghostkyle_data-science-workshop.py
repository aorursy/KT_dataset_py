# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split # get a model 

from sklearn.linear_model import LogisticRegression  # algorithm for training



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os 

print(os.listdir("../input"))

# This prints the files in the ./input/ folder



# get the goood data

train = pd.read_csv("../input/train.csv") 

train.info()

#lists the meta data

print("\n")

train.describe() 

# some info for each col
colsToDrop = ['Ticket', 'Name', 'Cabin', 'PassengerId']

train = train.drop(colsToDrop, axis=1)

print(train.shape)
train['Embarked'] = train['Embarked'].fillna('C')

# 2 unknowen if embared or not

train['Sex'] = train['Sex'].apply(lambda x: 1 if  x == 'female' else 0)

# populate gender with numbers 

train['Embarked'] = train['Embarked'].map({'S' : 0, 'Q' : 1, 'C': 2}).astype(int)

# change embarked status to numbers

train.head()
def fillAges(df):

    count = df['Age'].isnull().sum()

    avg = df['Age'].mean()

    std = df['Age'].std(); 

    

    #                          min         max       how many

    random_ages = np.random.randint(avg - std, avg + std, count)

    df['Age'][np.isnan(df['Age'])] = random_ages 

    return df 



#populate missing ages with random +/- 1 std of mean

train = fillAges(train)
sns.set(style="dark")



plt.figure(figsize=(10,12))

plt.title('HEAT (map)')

sns.heatmap(train.astype(float).corr(), annot=True)

#Heat map corrilations
y = train['Survived']              #  just survived

x = train.drop('Survived', axis=1) #  data after droping survived 



#80% train, 20% test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



lr = LogisticRegression()

lr.fit(x_train, y_train) # train the model 



#test the model 

score = round(lr.score(x_test, y_test) * 100, 2)

print(score)