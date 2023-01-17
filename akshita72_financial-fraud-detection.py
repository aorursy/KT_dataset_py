# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filename = '/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv'

data = pd.read_csv(filename)

data.head()
data.dtypes
data.isnull().sum()
data.size
perFraud = (data[data['isFraud']==1].size/data.size)*100

print(perFraud)
data.columns
predictors = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 

              'newbalanceDest', 'isFlaggedFraud']



XX = data[predictors]

X = pd.get_dummies(XX)  # one-hot-encoding

X.describe()
y = data.isFraud
train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)
clf = DecisionTreeClassifier()

clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
print(confusion_matrix(test_y, pred_y))

print(classification_report(test_y, pred_y))
sc = StandardScaler()

train_X = sc.fit_transform(train_X)

test_X = sc.transform(test_X)
clf = RandomForestClassifier(n_estimators=20, random_state=0)

clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
print(confusion_matrix(test_y,pred_y))

print(classification_report(test_y,pred_y))

print(accuracy_score(test_y,pred_y))