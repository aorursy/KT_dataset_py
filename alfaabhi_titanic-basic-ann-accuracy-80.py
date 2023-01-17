import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning) # This is because we will be using make some visualizating with features having missing values



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
raw_path = '/kaggle/input/titanic/'

train = pd.read_csv(raw_path + "train.csv", dtype={'Survived': str})

print(train.shape)

test = pd.read_csv(raw_path + "test.csv")

print(test.shape)

test_train = pd.concat([train, test], sort=False)

print(test_train.shape)
test_train.head()
test_train.info()
sns.pairplot(test_train[~test_train['Survived'].isnull()], hue='Survived')
sns.countplot('Sex', data=test_train[~test_train['Survived'].isnull()], hue='Survived') # you can directly use the dataframe train
test_train.head()
test_train['Title'] = test_train['Name'].apply(lambda x: x.split('.')[0].split()[-1].strip().upper())

test_train['Title'].unique()
test_train[test_train['Title'] == "COUNTESS"]
plt.figure(figsize=(18,10))

sns.set_style('darkgrid')

sns.countplot('Title', data=test_train[~test_train['Survived'].isnull()], hue='Survived')
test_train[test_train['Age'].isnull()].shape
avg_age_title = test_train.groupby(['Title'])['Age'].mean().reset_index()

avg_age_title
for title in list(test_train[test_train['Age'].isnull()]['Title'].unique()):

    test_train.loc[(test_train['Age'].isnull()) & (test_train['Title'] == title), 'Age'] = avg_age_title[avg_age_title['Title'] == title]['Age'].unique()[0]
test_train[test_train['Age'].isnull()].shape
test_train['AgeBin'] = pd.cut(test_train['Age'].astype(int), 5)
sns.countplot("AgeBin", data=test_train[~test_train['Survived'].isnull()], hue='Survived')
test_train[test_train['Embarked'].isnull()].shape
test_train['Embarked'].fillna(test_train['Embarked'].mode()[0], inplace = True)

test_train[test_train['Embarked'].isnull()].shape
sns.countplot("Embarked", data=test_train[~test_train['Survived'].isnull()], hue='Survived')
test_train['Companions'] = test_train['SibSp'] + test_train['Parch']

test_train['Loner'] = 1

test_train.loc[test_train['Companions'] > 0, 'Loner'] = 0
test_train.info()
test_train[test_train['Fare'].isnull()]
impute_fare = test_train.groupby(['AgeBin', 'Embarked', 'Sex'])['Fare'].mean().reset_index()

impute_fare[(impute_fare['Embarked'] == 'S') & (impute_fare['Sex'] == 'male')]
test_train['Fare'].fillna(38.41, inplace=True)

test_train['FareBin'] = pd.qcut(test_train['Fare'], 5)
min_req = 8

title_names = (test_train['Title'].value_counts() < min_req)

test_train['Title'] = test_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(test_train['Title'].value_counts())
test_train.drop(labels=['Age', 'Fare', 'Name', 'Ticket', 'PassengerId', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

test_train.info()
label = LabelEncoder()  

test_train['Sex_'] = label.fit_transform(test_train['Sex'])

test_train['Embarked_'] = label.fit_transform(test_train['Embarked'])

test_train['Title_'] = label.fit_transform(test_train['Title'])

test_train['AgeBin_'] = label.fit_transform(test_train['AgeBin'])

test_train['FareBin_'] = label.fit_transform(test_train['FareBin'])

test_train.head()
test_train.drop(labels=['Sex', 'Embarked', 'Sex', 'AgeBin', 'FareBin', 'Title'], axis=1, inplace=True)
frame = pd.get_dummies(test_train, prefix=['Pclass', 'Embarked', 'Title', 'AgeBin', 'FareBin'], columns=['Pclass', 'Embarked_', 'Title_', 'AgeBin_', 'FareBin_'], drop_first=True)

frame.head()
frame.info()
plt.figure(figsize=(18, 10))

sns.heatmap(frame[~frame['Survived'].isnull()].astype(float).corr(), cmap='RdBu', annot=True)
X_train = frame[0:891].drop('Survived', axis=1)

y_train = frame[0:891]['Survived']

X_test = frame[891:].drop('Survived', axis=1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
from keras.layers import Dense

from keras.models import Sequential
model = Sequential()



model.add(Dense(units=9, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))

model.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, epochs=60) # Almost converged at 50 epochs. You can change the epochs and batch size and see the changes.
y_pred = model.predict(X_test) # This will return the probabilities of a passenger surviving.
y_final = (y_pred > 0.59).astype(int).reshape(X_test.shape[0]) # This will convert the probabilities to flags, based on the thresold value.
# output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})

# output.to_csv('prediction.csv', index=False)