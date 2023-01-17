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
import seaborn as sb

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/ngc-628-7793-krumholz-2015/opencluster.tsv', delimiter = ';')
df = df.replace(r'^\s*$', np.nan, regex=True)
#we dont need them

df = df.drop(['Lib','beta','gamma'], axis = 1)
df['Mode'] = df['Mode'].astype(int)
# mode

plt.figure(figsize=(12, 7))

sb.set_style("whitegrid")

sb.countplot(x = 'Mode', data = df)
plt.figure(figsize=(12, 7))

sb.distplot(df['logM-84'])

df['logM-84'].describe()
plt.figure(figsize=(12, 7))

sb.distplot(df['logT-84'])

df['logT-84'].describe()
for i in range (0,5):

    classes = df.loc[(df["Mode"]) == i]

    print('Mode ',i)

    print(classes['logM-84'].describe(), '\n')
plt.figure(figsize=(12, 7))

sb.boxplot(x='Mode',y='logM-84',data=df)
plt.figure(figsize=(12, 7))

sb.boxplot(x='Mode',y='logT-84',data=df)
from sklearn.model_selection import train_test_split



dataset = df

dataset.drop(['Field','ID'],axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('Mode',axis=1), 

                                                    df['Mode'], test_size=0.25, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
regressor = LogisticRegression()

regressor.fit(X_train, y_train)

log_pred = regressor.predict(X_test)



print('accuracy score: ', accuracy_score(y_test, log_pred),'\n')

print(classification_report(y_test, log_pred))

print(confusion_matrix(y_test, log_pred))

print(accuracy_score(y_test, log_pred))
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier



sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

print('accuracy score: ', accuracy_score(y_test, sgd_pred),'\n')

print(classification_report(y_test, sgd_pred))

print(confusion_matrix(y_test, sgd_pred))

print(accuracy_score(y_test, sgd_pred))