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
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head(10)
hours = df['Time']/3600

hours = hours.astype(int)

df['Hours'] = hours
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize= (15,5))

sns.set(style="whitegrid")

sns.countplot(x='Hours',data = df , hue = 'Class',palette='BuPu')

plt.title("Graph of Transactions per each hour\n", fontsize=16)

sns.set_context("paper", font_scale=1.4)



plt.show()
a= len(df[df['Class'] == 0] )

print ("Amount of Non Fraud transactions = " , a)
b = len(df[df['Class'] == 1])

print ("Amount of Fraud transactions = " ,b )
ratio = [ a, b] 

title = "Not Fraud" , "Fraud"



plt.figure(figsize=(9,9))

plt.pie(ratio, labels= title, shadow=True, startangle=0)

plt.title('Pie Chart Ratio of Transactions by their Class\n', fontsize=16)

sns.set_context("paper", font_scale=1.2)



from sklearn.model_selection import train_test_split

from sklearn.utils import resample



Y = df.Class

X = df.drop(['Time','Class', 'Amount'], axis=1)



# setting up testing and training sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2727)



# concatenate our training data back together

X = pd.concat([X_train, Y_train], axis=1)



notFraud = X[X.Class==0]

fraud = X[X.Class==1]
not_fraud_downsampled = resample(notFraud,

                                 replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled.Class.value_counts()
sns.countplot('Class', data=downsampled)

plt.title('Equally Distributed Classes', fontsize=14)

plt.ylabel("Frequency")

plt.show()
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

Y_train = downsampled.Class

X_train = downsampled.drop('Class', axis=1)



undersampled = XGBClassifier()

undersampled.fit(X_train, Y_train)



# Predict on test

undersampled_pred = undersampled.predict(X_test)

# predict probabilities

probs = undersampled.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]



accuracy = accuracy_score(Y_test, undersampled_pred)

print(accuracy)
from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(X_train, Y_train)

prediction=log_model.predict(X_test)

score= accuracy_score(Y_test, prediction)

print(score)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 100)



clf.fit(X_train , Y_train)





pred=clf.predict(X_test)

sc= accuracy_score(Y_test, pred)

print(sc)