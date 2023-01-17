# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

Test_PIDs = test["PassengerId"]

train.head()
train = train.drop(["Name","Ticket","Cabin","PassengerId"], axis=1)

test = test.drop(["Name","Ticket","Cabin","PassengerId"], axis=1)
print(train.isnull().sum())

print(test.isnull().sum())
# print(train.Survived.value_counts())

# print(train.Pclass.value_counts().sort_index())

# print(train.Sex.value_counts())

# print(train.Embarked.value_counts().sort_values())
# train = train.fillna(train.mean())

# test = test.fillna(test.mean())

train = train.fillna(train.median())

test = test.fillna(test.median())
train.loc[train["Embarked"].isnull(),"Embarked"]="S"
print(train.isnull().sum())

print(test.isnull().sum())
train.head()
test.head()
#train.Survived.hist()

#train.Pclass.hist()
#train.Sex.hist()

#train.Embarked.hist()
train = pd.get_dummies(train, columns=['Sex','Embarked','Pclass'], drop_first=True)

test = pd.get_dummies(test, columns=['Sex','Embarked','Pclass'], drop_first=True)
train.head()
test.head()
#train['Age'].describe()
X = train.drop(["Survived"], axis=1)

Y = train[["Survived"]]
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=2019)
#logreg = LogisticRegression()

logreg = LogisticRegression(C=1e3,solver='lbfgs',random_state=2019)

logreg.fit(x_train, y_train)
print(logreg.intercept_)

print(logreg.coef_)
#y_pred = logreg.predict(x_val)

y_pred = logreg.predict_proba(x_val)[:, 1]
THRESHOLD=0.66789

y_pred_binary = (y_pred > THRESHOLD).astype("int").ravel()
metrics.accuracy_score(y_val, y_pred_binary)
metrics.roc_auc_score(y_val, y_pred_binary)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred_binary)

plt.plot(fpr,tpr,label="data 1, auc="+str(metrics.roc_auc_score(y_val, y_pred_binary)))
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_val, y_pred_binary)), annot=True, cmap="YlGnBu", fmt='g')
#y_pred = logreg.predict(test)

y_pred = logreg.predict_proba(test)[:, 1]
y_pred_binary = (y_pred > THRESHOLD).astype("int").ravel()

y_pred_binary
submission = {}

submission['PassengerId'] = Test_PIDs

submission['Survived'] = y_pred_binary



submission = pd.DataFrame(submission)



submission = submission[['PassengerId', 'Survived']]

submission = submission.sort_values(['PassengerId'])

submission.to_csv("submisision.csv", index=False)
print(submission['Survived'].value_counts())