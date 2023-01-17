# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.layers import Input, Dense

from keras.models import Model, Sequential

from keras import regularizers

from keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

from sklearn import preprocessing 

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np

import re as re



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train, test]



# Any results you write to the current directory are saved as output.
train.head()
print (train.info())
# length of name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)



# Has Cabin 

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
print (train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)



print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)



    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping FamilySize

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

    

    

# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



X = train.drop(["Survived"], axis=1)

y = train["Survived"].values
model = Sequential()



## encoding part

model.add(Dense(100, input_shape=(X.shape[1],), activation='tanh', activity_regularizer=regularizers.l1(10e-5)))

model.add(BatchNormalization())

model.add(Dense(50, activation='relu'))





## decoding part

model.add(Dense(50, activation='tanh'))

model.add(BatchNormalization())

model.add(Dense(100, activation='relu'))





model.add(Dense(X.shape[1], activation='relu'))



model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
model.summary()
# Before training, let's perform min max scaling.

scaler = preprocessing.MinMaxScaler()

scaler.fit(X.values)

X_scale = scaler.transform(X.values)

test_x_scale = scaler.transform(test.values)



x_perished, x_survived = X_scale[y == 0], X_scale[y == 1]

history = model.fit(x_perished, x_perished, epochs = 20, shuffle = True, validation_split = 0.2)
train_result = pd.DataFrame(history.history)



style = {'loss': 'go--', 'val_loss': 'bo--'}

train_result[['loss', 'val_loss']].plot(style=style)
hidden_representation = Sequential()

hidden_representation.add(model.layers[0])

hidden_representation.add(model.layers[1])

hidden_representation.add(model.layers[2])
perished_hid_rep = hidden_representation.predict(x_perished)

survived_hid_rep = hidden_representation.predict(x_survived)



rep_x = np.append(perished_hid_rep, survived_hid_rep, axis = 0)

y_n = np.zeros(perished_hid_rep.shape[0])

y_f = np.ones(survived_hid_rep.shape[0])

rep_y = np.append(y_n, y_f)
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.2)

clf = LogisticRegression().fit(train_x, train_y)

pred_y = clf.predict(val_x)



print (classification_report(val_y, pred_y))

print (accuracy_score(val_y, pred_y))
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

test_rep_x = hidden_representation.predict(test_x_scale)

temp['Survived'] = [int(x) for x in clf.predict(test_rep_x)]

temp.to_csv("submission.csv", index = False)

temp.head()