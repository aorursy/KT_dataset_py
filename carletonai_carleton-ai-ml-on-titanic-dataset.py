# data structure manipulation libraries

import numpy as np

import pandas as pd



# data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns 



# data splitting

from sklearn.model_selection import train_test_split



# machine learning algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



import os

print(os.listdir('../input'))
train = pd.read_csv('../input/train.csv')

train.head()
train.info()
train.describe()
# The number of empty values for each column

cols = list(train)

for i in cols:

    print(i, ' '*(20-len(i)), train[i].isnull().sum())
# Dropping irrelevant (for this demo) columns 



print('Before: ', train.shape)



cols_to_drop = ['Ticket', 'Name', 'Cabin', 'PassengerId']

train = train.drop(cols_to_drop, axis=1)



print('After:  ', train.shape)
# fill empty embarkation values

train['Embarked'] = train['Embarked'].fillna('C')



# transforming text data into numerical values

train['Sex'] = train['Sex'].apply(lambda x: 1 if x=='female' else 0)

train['Embarked'] = train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

train.head()
# fill empty ages of passengers according to the distribution of existing ages data

def fillAges(df):

    count = df['Age'].isnull().sum()

    avg = df['Age'].mean()

    std = df['Age'].std()

    random_age = np.random.randint(avg - std, avg + std, count)

    df['Age'][np.isnan(df['Age'])] = random_age

    return df



train = fillAges(train)



# def groupAges(df):

#     df.loc[df['Age'] <= 18, 'Age'] = 0

#     df.loc[(df['Age'] > 18) & (df['Age'] <= 36), 'Age'] = 1

#     df.loc[(df['Age'] > 36) & (df['Age'] <= 54), 'Age'] = 2

#     df.loc[(df['Age'] > 54) & (df['Age'] <= 72), 'Age'] = 3

#     df.loc[df['Age'] > 72, 'Age'] = 4

#     return df



# train = groupAges(train)



train.head()
plt.figure(figsize=(10, 12))

plt.title('Heatmap (Correlation between columns)')

colormap = plt.cm.RdBu

sns.heatmap(train.astype(float).corr(), annot=True, cmap=colormap)
# An array of 'Survived' data of the passengers. We call these 'labels' of the passengers.

# The label is what our model is going to predict

y = train['Survived']



# Passenger data without the 'Survived' data. We call these the 'features' of the passengers.

# Our model will base its prediction of whether a passenger survived based on his or her features

x = train.drop('Survived', axis=1)



# Think of x_train as a sample exam, and y_train the answers to the sample exam.

# Think of x_test as the final exam, and y_test the answers to the final exam.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# set up our training algorithm

lr = LogisticRegression()



# train our model using the 'sample exam'

lr.fit(x_train, y_train)



# score our model using the 'final exam'

lr_score = round(lr.score(x_test, y_test) * 100, 2)



# the score represents the percentage of the survival data our model predicts correctly

print(lr_score)
# Let's trying training our model using another algorithm

svc = SVC()

svc.fit(x_train, y_train)



svc_score = round(svc.score(x_test, y_test) * 100, 2)

print(svc_score)