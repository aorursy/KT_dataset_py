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

        

import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.isna().sum()
train_data.head()
y=train_data["Survived"]

features=["Pclass","Sex","SibSp","Parch"]

X=pd.get_dummies(train_data[features])

X_test=pd.get_dummies(test_data[features])



#model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)

#model.fit(X, y)

from sklearn.linear_model import LogisticRegression

model2=LogisticRegression()

model2.fit(X,y)

predictions = model2.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

X_train,X_test80,Y_train,Y_test=train_test_split(X,y,test_size=.25,random_state=5)

model1=RandomForestClassifier(n_estimators=200,max_depth=5,random_state=1)

model1.fit(X_train,Y_train)

prediction1=model1.predict(X_test80)

score_random_forest=sklearn.metrics.accuracy_score(Y_test,prediction1)



from sklearn.linear_model import LogisticRegression

model2=LogisticRegression()

model2.fit(X_train,Y_train)

prediction2=model2.predict(X_test80)

score_logistic_regression=sklearn.metrics.accuracy_score(Y_test,prediction2)



from sklearn.ensemble import GradientBoostingClassifier

model3=GradientBoostingClassifier()

model3.fit(X_train,Y_train)

prediction3=model3.predict(X_test80)

score_gradient_boosting=sklearn.metrics.accuracy_score(Y_test,prediction3)



print("Random forest score ", score_random_forest)

print("Logistic regression score ",score_logistic_regression)

print("Gardient boosting score ",score_gradient_boosting)
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



cm=confusion_matrix(Y_test,prediction2)

conf_mat=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize=(8,5))

sns.heatmap(conf_mat,annot=True,fmt='d')