# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.head()
data['Sex1'] = [0 if i == "female" else 1 for i in data.Sex]

data['Embarked1'] = [0 if i == "S" else 1 if i=='C' else 2 for i in data.Embarked]

data.head()
data.info()
data = data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)

data.dropna(inplace=True)
x,y = data.loc[:,data.columns != 'Survived'], data.loc[:,'Survived']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
x_train.head()
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report
logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)

#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
from sklearn.metrics import accuracy_score

print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
data_set = pd.read_csv("../input/test.csv")

data_set['Sex1'] = [0 if i == "female" else 1 for i in data_set.Sex]

data_set['Embarked1'] = [0 if i == "S" else 1 if i=='C' else 2 for i in data_set.Embarked]

data_set.info()
data_result = data_set[['PassengerId']]

data_set = data_set.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)

data_set = data_set.fillna(data_set.mean())
data_set.info()
y_pred = logreg.predict(data_set)
len(y_pred)
data_result['Survived'] = y_pred

data_result.head()
data_result.to_csv('submission.csv',index=False)