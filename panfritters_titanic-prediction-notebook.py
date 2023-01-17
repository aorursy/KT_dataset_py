# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

from sklearn import linear_model

from sklearn.naive_bayes import GaussianNB

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

figure_size = (12, 12)

# Any results you write to the current directory are saved as output.

%matplotlib inline
df_gender = pd.read_csv('../input/gender_submission.csv')

df_gender.head()

df_gender.describe()

df_gender.hist()

df_train = pd.read_csv('../input/train.csv')

df_train.describe()
df_train.hist(figsize=figure_size, bins=30)
df_train.head()
#Get Rid of unnessesary columns

df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
len(df_train)
df_train.head()
display(df_train.isna().sum())
dm_df = pd.get_dummies(df_train, prefix='is_')

dm_df.head()

display(dm_df.isna().sum())

dm_df = dm_df.dropna()

display(dm_df.isna().sum())
len(dm_df)


correllations = dm_df.corr()

display(dm_df)

fig = plt.figure(figsize=figure_size)

ax = fig.add_subplot(111)

cax = ax.matshow(correllations, vmin=-1, vmax=1)

fig.colorbar(cax)

names = dm_df.columns

ticks = np.arange(0,len(names),1)

display(names)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()

len(dm_df)
labels = dm_df['Survived']

dm_df.drop(['Survived'], inplace=True, axis=1)
len(dm_df)



X_train, X_test, y_train, y_test = train_test_split(dm_df, labels, test_size=0.5)



clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

test_pred = clf.predict(X_test)



print(classification_report(y_test, test_pred))

accuracy_score(y_test, test_pred)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

regr_pred = logreg.predict(X_test)

print(classification_report(y_test,regr_pred))

accuracy_score(y_test, regr_pred)
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

clf.fit(X_train,y_train)

sgd_preds = clf.predict(X_test)

print(classification_report(y_test,sgd_preds))

accuracy_score(y_test, sgd_preds)
clf = GaussianNB()

clf.fit(X_train, y_train)

gauspreds = clf.predict(X_test)

print(classification_report(y_test,gauspreds))

accuracy_score(y_test, gauspreds)
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim=10, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model