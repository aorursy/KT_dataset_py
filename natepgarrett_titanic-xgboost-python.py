# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

from matplotlib import pyplot

from sklearn.model_selection import KFold, train_test_split, GridSearchCV

#from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score

from sklearn.datasets import load_iris, load_digits, load_boston

from fancyimpute import KNN   

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
train.head()
train.isnull().sum()
X = pd.concat([train.iloc[:,2],train.iloc[:,4:12]], axis=1)

y = train.iloc[:,1]

#Sex, Ticket, Cabin, Embarked need to be dummied, maybe Pclass to

tempdummy = pd.get_dummies(X['Sex'])

tempX = pd.concat([X,tempdummy],axis=1)

del tempX['Sex']

del tempX['Ticket'] #not a good column

del tempX['Cabin'] #not a good column

tempdummy = pd.get_dummies(tempX['Embarked'])

finalX = pd.concat([tempX,tempdummy],axis=1)

del finalX['Embarked']

finalX.head()



# fit model no training data

model = XGBClassifier()

model.fit(finalX, y)



# plot

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
tempdummy = pd.get_dummies(finalX['Pclass'])

newX = pd.concat([finalX,tempdummy],axis=1)

del newX['Pclass']

newX.head()

model = XGBClassifier()

model.fit(newX, y)



# plot

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
del newX['male']

del newX['S']

del newX[3]

newX.head()

model = XGBClassifier()

model.fit(newX, y)



# plot

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
model.feature_importances_
newX.isnull().sum()

X_filled_knn = KNN(k=5).fit_transform(newX)





model = XGBClassifier()

model.fit(X_filled_knn, y)



# plot

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()
model.feature_importances_


X_train, X_test, y_train, y_test = train_test_split(X_filled_knn, y, test_size = 0.25, random_state = 0)

classifier = XGBClassifier()

classifier.fit(X_train,y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

cm



roc_auc_score(y_test,y_pred)

#'''

#y = train[,1]

#X = train['data']

#xgb_model = xgb.XGBRegressor()

#clf = GridSearchCV(xgb_model,

#                   {'max_depth': [2,4,6],

#                    'n_estimators': [50,100,200]}, verbose=1)

#clf.fit(X,y)

#print(clf.best_score_)

#print(clf.best_params_)

#'''
test = pd.read_csv('../input/test.csv')

test.head()



testX = pd.concat([test.iloc[:,1],test.iloc[:,3:12]], axis=1)

testX.head()

tempdummy = pd.get_dummies(testX['Sex'])

testtempX = pd.concat([testX,tempdummy],axis=1)

del testtempX['Sex']

del testtempX['Ticket'] #not a good column

del testtempX['Cabin'] #not a good column

tempdummy = pd.get_dummies(testtempX['Embarked'])

testfinalX = pd.concat([testtempX,tempdummy],axis=1)

del testfinalX['Embarked']

tempdummy = pd.get_dummies(testfinalX['Pclass'])

testnewX = pd.concat([testfinalX,tempdummy],axis=1)

del testnewX['Pclass']

del testnewX['male']

del testnewX['S']

del testnewX[3]

testX_filled_knn = KNN(k=5).fit_transform(testnewX)
test_preds = classifier.predict(testX_filled_knn)

testX = pd.concat([test.iloc[:,0],pd.DataFrame(test_preds)], axis=1)

testX.columns = ['PassengerId','Survived']

#testX.head()

testX.to_csv('finalpreds.csv',index=False)