import numpy as np

import pandas as pd

import pylab as pl



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# Top of the training data

#train.head(3)



# Summary statistics of training data

train.describe()



#train[train['Age'] > 60][['Age','Sex','Pclass','Survived']]

#train[train['Age'].isnull()][['Age','Sex','Pclass','Survived']]
for i in range(1,4):

    print(i,len(train[(train['Sex']=='male') & (train['Pclass']==i)]))
train['Age'].dropna().hist(bins=16, alpha=.5)
# Creating a new column in df train

train['Gender'] = 4



# Assigning M or F in Gender column instead of male or female in Sex

#train['Gender'] = train['Sex'].map(lambda x: x[0].upper())



#Assingning 0 or 1 in Gender column instead of male or female in Sex

train['Gender'] = train['Sex'].map({'female':0,'male':1}).astype(int)

train.head(20)
# rows for gender and columns for class

median_ages = np.zeros((2,3))

for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j] = train[(train['Gender'] == i) & (train['Pclass'] == j+1)].dropna().median()

median_ages 