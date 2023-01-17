# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')

test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')
X = train_df.drop(['Id', 'label','Soil'], axis=1)

y = train_df['label']

# Splitting the dataset into the Training set and Test set

#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 0)

#from imblearn.over_sampling import SMOTE

# Feature Scaling

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SVMSMOTE

sc = StandardScaler()

X = sc.fit_transform(X)

sm = SVMSMOTE(random_state=2)

X, y = sm.fit_sample(X, y.ravel())

#print(len(X_train_2), len(X_train))

# Fitting Random Forest Classification to the Training set

#from sklearn.ensemble import RandomForestClassifier

#classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', class_weight={-1 : 0.1, 1:10})

#classifier.fit(X_train, y_train)

#classifier1 = RandomForestClassifier(n_estimators = 80, criterion = 'entropy')

#classifier1.fit(X_train_2, y_train_2)
# X is the feature set and y is the target

#from sklearn.svm import SVC

#print("hi")

#classifier = SVC(kernel = 'rbf', gamma = 'auto',probability = True)

#print("hi")

import xgboost as xgb

classifier=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

from sklearn.model_selection import StratifiedKFold

rkf = StratifiedKFold(n_splits=30)

print("hi")

for train_index, test_index in rkf.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm1 = confusion_matrix(y_test, y_pred)

    print(cm1)

    

    



        
from sklearn.svm import SVC

classifier1 = SVC(kernel = 'rbf', gamma = 'auto', probability = True)

from sklearn.model_selection import StratifiedKFold

rkf = StratifiedKFold(n_splits=30)

for train_index, test_index in rkf.split(X,y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    classifier1.fit(X_train, y_train)

    y_pred = classifier1.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm2 = confusion_matrix(y_test, y_pred)

    print(cm2)
#pred1=classifier.predict_proba(X)

#pred2=classifier1.predict_proba(X)

#finalpred=(1.2*pred1+pred2)/2

#print(np.shape(finalpred))
'''test_res=[]

for i in range(len(finalpred)):

    if (finalpred[i][0] > finalpred[i][1]):

        test_res.append(-1)

    else:

        test_res.append(1)

'''
#print(np.shape(test_res))
test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv', header =0)

X_test = test_df.drop(['Id','Soil'], axis=1)

sc = StandardScaler()

X_test = sc.fit_transform(X_test)

#test_res = classifier.predict(X_test)

pred1=classifier.predict_proba(X_test)

pred2=classifier1.predict_proba(X_test)

finalpred=(pred1+pred2)/2

test_res=[]

for i in range(len(finalpred)):

    if (finalpred[i][0] > finalpred[i][1]):

        test_res.append(-1)

    else:

        test_res.append(1)
print(np.shape(test_res))
submission_df = pd.DataFrame()

submission_df['Id'] = test_df['Id']
submission_df['Predicted'] = test_res
submission_df.to_csv('ans.csv',index=False)