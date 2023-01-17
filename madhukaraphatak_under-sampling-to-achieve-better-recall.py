# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## Read the data

df = pd.read_csv("../input/creditcard.csv")

df.head()
## Plot the distribution of data

%matplotlib inline

sns.countplot(x='Class', data=df)
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split



df['normal_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df = df.drop(['Amount','Time'], axis=1)

X = df.loc[:,df.columns != 'Class']

y = df.loc[:,df.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# Calculate the recall score for logistic Regression on Skewed data

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score,accuracy_score

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(recall_score(y_test,y_pred,average=None))

print(accuracy_score(y_test,y_pred))

# Undersample the data

no_frauds = len(df[df['Class'] == 1])

non_fraud_indices = df[df.Class == 0].index

random_indices = np.random.choice(non_fraud_indices,no_frauds, replace=False)

fraud_indices = df[df.Class == 1].index

under_sample_indices = np.concatenate([fraud_indices,random_indices])

under_sample = df.loc[under_sample_indices]
## Plot the distribution of data for undersampling

%matplotlib inline

sns.countplot(x='Class', data=under_sample)
X_under = under_sample.loc[:,under_sample.columns != 'Class']

y_under = under_sample.loc[:,under_sample.columns == 'Class']

X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.3, random_state = 0)
lr_under = LogisticRegression()

lr_under.fit(X_under_train,y_under_train)

y_under_pred = lr_under.predict(X_under_test)

print(recall_score(y_under_test,y_under_pred))

print(accuracy_score(y_under_test,y_under_pred))
## Recall for the full data

y_pred_full = lr_under.predict(X_test)

print(recall_score(y_test,y_pred_full))

print(accuracy_score(y_test,y_pred_full))
lr_balanced = LogisticRegression(class_weight = 'balanced')

lr_balanced.fit(X_train,y_train)

y_balanced_pred = lr_balanced.predict(X_test)

print(recall_score(y_test,y_balanced_pred))

print(accuracy_score(y_test,y_balanced_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix_value = confusion_matrix(y_test,y_balanced_pred)
sns.set(font_scale=1.4)

confusion_matrix_value

#sns.heatmap(confusion_matrix_value, annot=True)