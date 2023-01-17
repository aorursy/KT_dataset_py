import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

test_df = pd.read_csv("../input/titanic/test.csv")

train_df= pd.read_csv("../input/titanic/train.csv")

train_df=train_df.drop(['Name','Ticket','Fare','Cabin'],axis=1)

test_df=test_df.drop(['Name','Ticket','Fare','Cabin'],axis=1)

combine = [train_df, test_df]
train_df =train_df.fillna(method='ffill')

test_df=test_df.fillna(method='ffill')
train_df.isnull().sum()
test_df.isnull().sum()
from sklearn.preprocessing import LabelEncoder

train_df=train_df.apply(LabelEncoder().fit_transform)

test_df=test_df.apply(LabelEncoder().fit_transform)

train_df
test_df
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df

X_train.shape, Y_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Ypred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Ypred})

submission['PassengerId']=submission['PassengerId'].add(892)

submission
filename = 'Titanic Predictions1.csv'



submission.to_csv(filename,index=False)


