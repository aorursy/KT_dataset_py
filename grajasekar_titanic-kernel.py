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

titanic_gender=pd.read_csv("../input/gender_submission.csv")

test=pd.read_csv("../input/test.csv")

train=pd.read_csv("../input/train.csv")

from sklearn.neighbors import KNeighborsClassifier

feature_cols=['Pclass','Sex','Age']
#convert sex to dummy variables male=1 female=0
train[train.Sex=='male']=1
train[train.Sex=='female']=0
X_train=train[feature_cols]
y_train=train['Survived']

#fit KNN model
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#create testing variables
test[test.Sex=='male']=1
test[test.Sex=='female']=0
X_test=test[feature_cols]

y_pred=knn.predict(X_test)

#convert predictions to series
loo=y_pred.tolist() #from list to series
y_pred=pd.Series(loo)

test['Predicted Survival']=y_pred

submission=test[['PassengerId','Survival']]


#save as CSV
submission.to_csv('titanicsub.csv')


