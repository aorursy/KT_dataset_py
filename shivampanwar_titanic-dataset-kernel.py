# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

traindf=pd.read_csv('../input/train.csv')
testdf=pd.read_csv('../input/test.csv')
traindf.head()
traindf.dtypes
print ("Dataframe shape is {}".format(traindf.shape))
traindf.head()
traindf.drop(['Name','Ticket','Cabin','SibSp','PassengerId'],axis=1,inplace=True)
testdf.drop(['Name','Ticket','Cabin','SibSp','PassengerId'],axis=1,inplace=True)
print ("Train df shape is {}".format(traindf.shape))
print ("Test df shape is {}".format(testdf.shape))
traindf.head()
traindf.dtypes
y_train=traindf['Survived']
traindf.drop('Survived',axis=1,inplace=True)
traindf.fillna(traindf.mean(), inplace=True)
traindf.head()
traindf.dtypes
newtraindf=pd.get_dummies(traindf,columns=['Pclass','Sex','Embarked'])
# newtestdf=pd.get_dummies(testdf,columns=['Pclass','Sex','Parch','SibSp','Embarked'])
newtraindf.head()
# parameters = {'max_depth':[2, 3, 4, 5, 6], 'max_features':[2, 3,4]}
# rf = RandomForestClassifier()
# clf = GridSearchCV(rf, parameters)
# clf.fit(newtraindf, y_train)
# best_clf = clf.best_estimator_
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(newtraindf, y_train)
testdf.fillna(testdf.mean(), inplace=True)
newtestdf=pd.get_dummies(testdf,columns=['Pclass','Sex','Embarked'])
# prediction=clf.predict(newtestdf)
predictions = gbm.predict(newtestdf)
len(predictions)
demo=pd.read_csv('../input/gender_submission.csv')
demo.head()
d = {'PassengerId': np.arange(892,1310), 'Survived': predictions}
outputdf = pd.DataFrame(data=d)
# outputdf.to_csv('Outputfile.csv')

outputdf.to_csv('example.csv')
