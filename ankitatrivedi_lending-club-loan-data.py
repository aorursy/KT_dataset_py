# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/predicting-who-pays-back-loans/loan_data.csv')

data.head()
data.info()
data.describe()
data.head()
plt.figure(figsize = (10,6))

data[data['credit.policy'] == 1]['fico'].hist(bins = 30, color = 'red' , label = ('credit policy = 1'), alpha = 0.6)

data[data['credit.policy'] == 0]['fico'].hist(bins = 30, color = 'blue', label = ('credit policy = 0'), alpha = 0.6)

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize = (10,6))

data[data['not.fully.paid'] == 1]['fico'].hist(bins = 30, color = 'red' , label = ('Not Fully Paid = 1'), alpha = 0.6)

data[data['not.fully.paid'] == 0]['fico'].hist(bins = 30, color = 'blue', label = ('Not Fully Paid = 0'), alpha = 0.6)

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize = (10,6))

sns.countplot(x = 'purpose', hue = 'not.fully.paid', data = data , palette= 'Set1')
sns.jointplot(x = 'fico', y = 'int.rate', data = data)
plt.figure(figsize = (10,6))

sns.lmplot( x = 'fico' , y = 'int.rate', data = data , hue = 'credit.policy', col = 'not.fully.paid', palette= 'Set1')
cat_feat = ['purpose']

final_data = pd.get_dummies(data , columns = cat_feat , drop_first = True )

final_data.info()
final_data.head(2)
from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis =1)

y = final_data['not.fully.paid']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 101)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

pred = dt.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print (classification_report(y_test, pred))

print (confusion_matrix(y_test, pred))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train , y_train)

pred_r = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred_r))

print(confusion_matrix(y_test, pred_r))