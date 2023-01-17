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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.options.display.max_columns=30

plt.style.use('fivethirtyeight')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc

from sklearn.model_selection import train_test_split

import pandas_profiling as pf
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df_exp = train_df

test_df_exp = test_df

train_df.head()
pf.ProfileReport(train_df)
train_df.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace=True)
train_df.isnull().sum()
#The names with Master in them are kids so their ages are mostly below 12. So we fill the NA for these ages with mean of the Master named kids

train_df_kids = train_df[train_df['Name'].str.contains('Master')]

train_df_kids['Age'].fillna(train_df[train_df['Name'].str.contains('Master')]['Age'].mean(),inplace=True)

train_df_Others = train_df[train_df['Name'].str.contains('Master')==False]

train_df = pd.concat([train_df_kids, train_df_Others])

train_df.info()
train_df["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

sns.countplot(train_df['Embarked'])
#The Embarked field has a majority field as 'S'. So we will fill that field with 'S'

train_df["Embarked"].fillna('S', inplace=True)
train_df.isnull().sum()
#combine the variables SibSp & Parch

train_df['Family']=np.where((train_df["SibSp"]+train_df["Parch"])>0, 1, 0)

train_df.drop('SibSp', axis=1, inplace=True)

train_df.drop('Parch', axis=1, inplace=True)
train_df.drop('Name',1, inplace=True)

train_df.head()
#Changing the age to a categorical field

def Age_Categories(x):

    if x<=16:

        c = 1

    elif x>16 and x<=40:

        c=2

    else:

        c=3

    return c

train_df.Age = train_df.Age.apply(Age_Categories)

train_df.Age.value_counts()
train_df.groupby(['Age'],sort=True)['Survived'].count().plot(kind='bar')
train_df.groupby(['Sex'],sort=True)['Survived'].count().plot(kind='bar')
train_df = pd.get_dummies(train_df, columns=["Pclass","Embarked","Sex",'Age'])

train_df.head()
test_df.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace=True)

test_df_kids = test_df[test_df['Name'].str.contains('Master')]

test_df_kids['Age'].fillna(test_df[test_df['Name'].str.contains('Master')]['Age'].mean(),inplace=True)

test_df_Others = test_df[test_df['Name'].str.contains('Master')==False]

test_df = pd.concat([test_df_kids, test_df_Others])

test_df["Age"].fillna(test_df["Age"].median(skipna=True), inplace=True)

test_df["Embarked"].fillna('S', inplace=True)

test_df['Family']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 1, 0)

test_df.drop(['SibSp','Parch','Name'], axis=1, inplace=True)

test_df.Age = test_df.Age.apply(Age_Categories)

test_df = pd.get_dummies(test_df, columns=["Pclass","Embarked","Sex",'Age'])

test_df.head()
train_df_exp.groupby(['Embarked'])['Survived'].count().plot(kind='bar', figsize=(8,8))
#Correlation Matrix

plt.figure(figsize=(12,12))

sns.heatmap(train_df.corr(), annot=True)
#Feature Importance

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1, max_depth=10)

model.fit(train_df.drop('Survived',1),train_df.Survived)

features = train_df.drop('Survived',1).columns

importances = model.feature_importances_

indices = np.argsort(importances)[-10:]  # top 10 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
#We can remove the Embarked columns as it has less Importance

Selected_columns = ['Fare', 'Family', 'Pclass_3','Embarked_C', 'Embarked_S', 'Sex_female', 'Sex_male', 'Age_1', 'Age_3']

X = train_df[Selected_columns]

y = train_df['Survived']

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

X = sc1.fit_transform(X)

test_df = sc1.transform(test_df[Selected_columns])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

reg1 = LogisticRegression()

reg1.fit(X_train,y_train)

y_pred = reg1.predict(X_test)

y_pred_proba = reg1.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Accuracy Score - ',accuracy_score(y_test, y_pred))

print('Classification Report - ',classification_report(y_test, y_pred))

print('auc Score - ',auc(fpr, tpr))
from sklearn.ensemble import GradientBoostingClassifier

classifier3 = GradientBoostingClassifier(random_state=0, n_estimators=80, max_depth=10, learning_rate=0.5, loss='exponential')

classifier3.fit(X_train, y_train)

y_pred = classifier3.predict(X_test)

y_pred_proba = classifier3.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Accuracy Score - ',accuracy_score(y_test, y_pred))

print('Classification Report - ',classification_report(y_test, y_pred))

print('auc Score - ',auc(fpr, tpr))
from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Dense, Activation, Dropout



# create model

model = Sequential()

model.add(Dense(units=128, input_dim=X.shape[1], kernel_initializer='normal', bias_initializer='zeros'))

model.add(Activation('relu'))

model.add(Dense(units=128, kernel_initializer='normal',bias_initializer='zeros'))

model.add(Activation('relu'))

model.add(Dropout(.2))

model.add(Dense(units=128, kernel_initializer='normal',bias_initializer='zeros'))

model.add(Activation('relu'))

model.add(Dropout(.2))

model.add(Dense(units=64, kernel_initializer='normal',bias_initializer='zeros'))

model.add(Activation('relu'))

model.add(Dropout(.2))

model.add(Dense(units=32, kernel_initializer='normal',bias_initializer='zeros'))

model.add(Activation('relu'))

model.add(Dropout(.2))

model.add(Dense(units=2))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y = pd.get_dummies(y)
model.fit(X, y.values, epochs=100,batch_size=32, verbose=2)
test_survived = model.predict_classes(test_df)

test_survived
submission = pd.DataFrame()

submission['PassengerId'] = test_df_exp.index

submission['Survived'] = test_survived

submission.to_csv("submission.csv", index=False)



submission.head(10)