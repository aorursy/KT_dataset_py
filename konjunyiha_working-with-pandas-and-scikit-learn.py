%matplotlib inline
import numpy as np
import pandas as pd
import random
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
%matplotlib inline

#Read train and test files
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# Look deep into data
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

# To see data in a more straightforward way
plt.figure()
train['Age'].dropna().plot(kind='hist', normed=True)
plt.title('Passenger age distribution')
plt.xlabel('Age')
plt.ylabel('Amount')
print('\n\nSurvived percentage in different Pclass:')
plt.figure()
plt.title('Passenger class distribution')
train['Pclass'].dropna().plot(kind='hist', normed=True)
plt.xlabel('Class')
plt.ylabel('Amout')

# Try some pic learn from other tutorial
plt.figure()
train_male = train.Survived[train.Sex == 'male'].value_counts().sort_index()
train_male.plot(kind='barh', label='Male', alpha=0.55, color='b')
train_female = train.Survived[train.Sex == 'female'].value_counts().sort_index()
train_female.plot(kind='barh', label='female', alpha=0.55, color='r')
plt.legend(loc='best')
# Data cleaning and feature selecting
sex_gender_map = {'female':0, 'male':1}
train['Gender'] = train['Sex'].map(sex_gender_map).astype(int)

## For each Gender and Pclass, calculate median age
train['AgeFill'] = train['Age']
median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                              (train['Pclass'] == j+1)]['Age'].median()
        train.loc[(train['Age'].isnull()) & (train['Gender'] == i) & (train['Pclass'] == j+1),\
                 'AgeFill'] = median_ages[i,j]
print('\n\nMedian ages:\n %s' %(median_ages))
#print(train[train['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10))
train['AgeIsNull'] = pd.isnull(train['Age']).astype(int)

print("\n\nSummary statistics of training data after data cleaning")
print(train.describe())