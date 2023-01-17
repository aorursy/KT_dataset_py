import pandas as pd

import numpy as np



train_path = '../input/train.csv'

test_path = '../input/test.csv'



train = pd.read_csv(train_path, index_col = 'PassengerId' )

test= pd.read_csv(test_path, index_col = 'PassengerId')

train.head()
import matplotlib.pyplot as plt

import seaborn as sns



def bar_plot(cat):

    live = train[train['Survived'] == 1][cat].value_counts()

    dead = train[train['Survived'] == 0][cat].value_counts()

    df = pd.DataFrame([live, dead, live/dead],['live', 'dead', 'live/dead'])

    print(df)

    df.plot.bar(stacked = True)
# number of male are more dead than number of female

bar_plot('Sex')
# Class 1 people having more survival chance than class 3



bar_plot('Pclass')
# Siblings and Spouse

bar_plot('SibSp')
# Parent and Child

bar_plot('Parch')
# Embarked

bar_plot("Embarked")
train.info()
# Age 177  , Cabin 687

train.isnull().sum()
# Age 87, Cabin 327

test.isnull().sum()
# fill the "Age" missing values in train and test dataset 

combined = [train, test]

for df in combined:

    df['Name']=df['Name'].str.extract(pat = '([\w]+)\.', expand=True)   
train['Name'].head()
map_title = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for df in combined:

    df['Name']=df['Name'].map(map_title)
train.head()
# fill missing 'Age' values 

train['Age'].fillna(train.groupby('Name')['Age'].transform('median'),inplace=True)

test['Age'].fillna(test.groupby('Name')['Age'].transform('median'),inplace=True)
sex_map = {'male': 0, 'female': 1}

for df in combined:

    df['Sex'] = df['Sex'].map(sex_map)

    
for df in combined:

    df['Age'] = np.where(df['Age'] <= 14, 0, df['Age'])

    df['Age'] = np.where((df['Age'] >14) & (df['Age'] <= 24), 1 ,df['Age'])

    df['Age'] = np.where((df['Age'] > 24) & (df['Age'] <=64 ), 2, df['Age'])

    df['Age'] = np.where(df['Age'] >64, 3, df['Age'])
train.head()
train['Cabin'].value_counts()
train.sample(40).plot.scatter(x='Pclass', y= 'Fare')