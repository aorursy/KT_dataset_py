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


train1=pd.read_csv('../input/train.csv')

test1=pd.read_csv('../input/test.csv')

test=pd.read_csv('../input/test.csv')



# splitting the Name column to get the title in a separate column,

tt=train1.Name.str.split(pat=',', expand=True)

tt.columns=['second','full']

titl=tt.full.str.split(pat='.', expand=True)

titl.columns=['t','f','n']

train1['title']=titl['t']





testt=test1.Name.str.split(pat=',', expand=True)

testt.columns=['second','full']

titl2=testt.full.str.split(pat='.', expand=True)

titl2.columns=['t2','f2']

test1['title']=titl2['t2']





# deleting columns with low importance,



train2=train1.drop(train1[['Name','PassengerId','Cabin','Ticket','Survived']], axis=1)



# encoding categorical columns,



train3=pd.get_dummies(train2)



test2=pd.get_dummies(test1)



# getting the common columns in training and test after encoding,



common=[]

for i in train3.columns:

    if i in test2.columns:

        common.append(i)



train3=train3[common]

test3=test2[common]



#filling blank fields,



from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)



train4=imp.fit_transform(train3)

train5=pd.DataFrame(train4)

train5.columns=train3.columns



# prepared data for processing,



X=train5

y=train1['Survived']





#splitting the training data to train/test to evaluate the model performance on the test set,



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.2)



# rescaling columns,



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled_tr=pd.DataFrame(scaler.fit_transform(X_train))

X_scaled_te=pd.DataFrame(scaler.fit_transform(X_test))



from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score



# fitting the model,

supvc = SVC(kernel = 'rbf',probability=True)

par= {'gamma':[0.1,0.5, 1,1.5,5],'C':[1,5,6,7]}

clf= GridSearchCV(supvc, param_grid = par, scoring = 'roc_auc', cv=5)

clf.fit(X_scaled_tr, y_train)

ydf=clf.decision_function(X_scaled_te)

score= roc_auc_score(y_test, ydf)
# filling na values in test set and rescaling the columns,



test4=imp.fit_transform(test3)

test5=pd.DataFrame(scaler.fit_transform(test4))



# Using the probability to get the survival,



yprob=clf.predict_proba(test5)[:,1]

results=[]

for i in yprob:

    if i >=0.5:

        results.append(1)

    else:

        results.append(0)

res=pd.DataFrame(results)

res.columns=['Survived']



res.index=test['PassengerId']

res.index.rename('PassengerId', inplace=True)

res.to_csv('12-6Aprsub.csv')