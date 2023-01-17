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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
one_hot=pd.get_dummies(train['color'])

train=train.drop('color',axis=1)

train=train.join(one_hot)
one_hot=pd.get_dummies(test['color'])

test=test.drop('color',axis=1)

test=test.join(one_hot)
import numpy as np



length=len(train['type'])

train['Type']=pd.Series(np.random.randn(length),index=train.index)

train['Type']=np.where(train['type']=='Ghoul',1,np.where(train['type']=='Goblin',2,3))

train=train.drop('type',axis=1)
train.head()
train.drop('Type',axis=1).apply(lambda x: x.corr(train.Type))



#The closer the value to 0, the lesser it is correlated to the target variable
#Checking for skewness in values

import matplotlib.pyplot as plt

%matplotlib inline

train.drop(['Type','black','blood','blue','clear','green','white','id'],axis=1).apply(lambda x: plt.hist(x))



#No significant skewness to account for
# Selecting features according to correlation,

# We can see only 'has_soul','bone_length','rotting_flesh'

# and 'hair_length' should be considered

features=['has_soul','bone_length','rotting_flesh','hair_length']

X=train[features]

Y=train.Type

X_test=test[features]
# Using GridSearch to select the best hyperparameters



from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

Cs = [0.00001,0.0001,0.001,0.01,0.1,1,10]

gammas=[0.0001,0.001,0.01,0.1,1]



param_grid = {'C': Cs,'gamma':gammas}

grid_search = GridSearchCV(SVC(), param_grid, cv=10)

grid_search.fit(X, Y)

grid_search.best_params_
model=SVC(C=1,gamma=0.1)
# Since we have less training data

# Let us use KFold Cross Validation

from sklearn.model_selection import KFold

kfold=KFold(n_splits=5)

print(kfold.get_n_splits(X))

print(kfold)
# Using accuracy_score as metric

from sklearn.metrics import accuracy_score



for train_index,test_index in kfold.split(X):

     X_train, X_val= X.iloc[train_index], X.iloc[test_index]

     Y_train, Y_val= Y.iloc[train_index], Y.iloc[test_index]

     model.fit(X_train,Y_train)

     predi=model.predict(X_val)

     print(accuracy_score(Y_val,predi))
pred=model.predict(X_test)
Id=np.array(test['id']).astype(int)

predictions=pd.DataFrame(pred,Id,columns=['Type'])



# Converting the numerical predictions back to String

length=len(predictions['Type'])

predictions['type']=pd.Series(np.random.randn(length),index=predictions.index)

predictions['type']=np.where(predictions['Type']==1,'Ghoul',np.where(predictions['Type']==2,'Goblin','Ghost'))

predictions=predictions.drop('Type',axis=1)
predictions.to_csv('output.csv',index_label=['id'])