# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
card = pd.read_csv("../input/creditcardfraud/creditcard.csv")

card.columns
card.isnull().any()
card['Class'].value_counts(normalize = True)
v_cols = card.drop(columns=['Time', 'Amount', 'Class'])

sns.boxplot(data=v_cols, palette="Set3")
fraud = card.loc[card['Class'] == 1]

no_fraud = card.loc[card['Class'] == 0]
sns.scatterplot(x="Time", y="Amount", data=fraud)
print("The most fraud done in the transaction over the past 2 days was {}".format(fraud.Amount.max()))

print("The average fraud done in the transaction over the past 2 days was {}".format(fraud.Amount.mean()))
sns.scatterplot(x="Time", y="Amount", data=no_fraud)
print("The most non-fraud done in the transaction over the past 2 days was {}".format(no_fraud.Amount.max()))

print("The average non-fraud done in the transaction over the past 2 days was {}".format(no_fraud.Amount.mean()))
X_var=card.drop(['Class'], axis=1)

y_var=card["Class"]

print(X_var.shape)

print(y_var.shape)

X=X_var.values

y=y_var.values
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.70, test_size=0.30, random_state=1)

card_model = DecisionTreeClassifier()

card_model.fit(X_train, y_train)

card_preds = card_model.predict(X_test)

print(mean_absolute_error(y_test, card_preds))
con_mat = confusion_matrix(y_test, card_preds)

con_mat
tp = con_mat[0][0]

fp = con_mat[0][1]

tn = con_mat[1][1]

fn = con_mat[1][0]

precision = (tp)/(tp+fp)

accuracy = (tp+tn)/(tp+tn+fp+fn)

sensitivity = (tp)/(tp+fn)

specificity = (tn)/(tn+fp)

recall_score = (tp)/(tp+fp)
print("Precision:", precision)

print("Accuracy:", accuracy)

print("Sensitivity:", sensitivity)

print("Specificity:", specificity)

print("Recall Score:", recall_score)