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
# EDA



df=pd.read_csv('../input/glass/glass.csv')

df.head()
df.shape
df.isna().sum()
df.describe
df.info()
df.dtypes
df['Type'].value_counts()
df.plot()
df.plot.hist()
df.plot.bar()
X = df.iloc[:,:-1].values

Y = df.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 177)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,accuracy_score



DT = DecisionTreeClassifier(criterion = 'entropy',splitter = 'best')

DT.fit(X_train,Y_train)



y_pred = DT.predict(X_test)

print(y_pred)

print("accuracy == {0}".format(accuracy_score(Y_test,y_pred)))
from xgboost import XGBClassifier



xboost = XGBClassifier(n_estimators = 500, learning_rate = 0.01)

xboost.fit(X_train,Y_train)



y_pred_X =xboost.predict(X_test)

print(y_pred_X)

print('Accuracy == {0}'.format(accuracy_score(Y_test,y_pred_X)))
from sklearn.ensemble import RandomForestClassifier



Rf = RandomForestClassifier(n_estimators = 500,random_state = 177)

Rf.fit(X_train,Y_train)



y_pred_R = Rf.predict(X_test)

print(y_pred_R)

print("accuracy == {0}".format(accuracy_score(Y_test,y_pred_R)))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)



knn_pred = knn.predict(X_test)

print(knn_pred)

print("accuracy == {0}".format(accuracy_score(Y_test,knn_pred)))
from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier(n_estimators=100,max_depth=5)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the train dataset

predict_train = model.predict(X_train)

# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)