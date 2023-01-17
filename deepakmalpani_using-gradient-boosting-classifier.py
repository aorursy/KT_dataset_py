# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

train['Sex']=train['Sex'].map({'male':0,'female':1})

train['Age']=train['Age'].fillna(np.mean(train['Age']));

train['Age']=train['Age'].astype(int)

train['Embarked']=train['Embarked'].map({'S':1,'C':0})

train['Embarked']=train['Embarked'].fillna(1)

train.head()
train[train['Embarked'].isna()].count()
X=train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

y=train['Survived']

print(X.shape)

print(y.shape)
from sklearn.manifold import TSNE

import seaborn as sns

model=TSNE(n_components=2,random_state=0)

tsne_data=model.fit_transform(X)

tsne_data=np.vstack((tsne_data.T,y)).T
tsne_df=pd.DataFrame(data=tsne_data,columns=['Dim1','Dim2','labels'])

sns.FacetGrid(tsne_df,size=6,hue='labels').map(plt.scatter,'Dim1','Dim2')

plt.legend()

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

print(knn.score(X_train,y_train))

print(knn.score(X_test,y_test))
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

n_estimators=[10,100]

min_samples_split=[2,3,4,5,6,7,8]

min_samples_leaf=[1,2,3,4,5,6,7,8]

max_features=[0.5,0.75,'auto']

params={'n_estimators':n_estimators,'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'max_features':max_features}

grd=GridSearchCV(model,param_grid=params)
#grd.fit(X_train,y_train)
#grd.best_score_
app_model=RandomForestClassifier(max_features=0.5,min_samples_leaf=1,min_samples_split=5,n_estimators=10)

app_model.fit(X_train,y_train)
app_model.score(X_train,y_train)
app_model.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()
n_estimators=[100,500,1000]

learning_rate=[0.01,0.1,1]

max_depth=[2,3,5]

params2={'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth}

grid_gbc=GridSearchCV(gbc,param_grid=params2)
grid_gbc.fit(X_train,y_train)
grid_gbc.best_params_
gbclf=GradientBoostingClassifier(learning_rate=0.01,max_depth=5,n_estimators=500)

gbclf.fit(X,y)

print(gbclf.score(X,y))

#print(gbclf.score(X_train,y_train))

#print(gbclf.score(X_test,y_test))
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(solver='lbfgs')

mlp.fit(X,y)

print(mlp.score(X,y))
test=pd.read_csv("../input/test.csv")

test['Sex']=test['Sex'].map({'male':0,'female':1})

test['Age']=test['Age'].fillna(np.mean(test['Age']));

test['Fare']=test['Fare'].fillna(np.mean(test['Fare']));

test['Age']=test['Age'].astype(int)

test['Embarked']=test['Embarked'].map({'S':1,'C':0})

test['Embarked']=test['Embarked'].fillna(1)

#test.head(10)
test_pid=test.PassengerId

x_test=test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]



print(x_test.shape)

#X_test[X_test['Fare'].isna()]

#print(y_test.shape)
y_pred2=gbclf.predict(x_test);

y_pred=mlp.predict(x_test)
s=pd.Series(dict(zip(list(test['PassengerId']),y_pred)))

d=dict(zip(list(test['PassengerId']),y_pred))

sub=pd.DataFrame(list(d.items()),columns=['PassengerID','Survived'])

#sub.reset_index()

#sub['Index']=pd.RangeIndex(start=0,stop=len(sub),step=1)

#sub.set_index('Index',inplace=True)

#sub

sub.to_csv('titanic3.csv',index=False)

#sub
submission=pd.read_csv('../input/gender_submission.csv')

y_test=submission['Survived']
gbclf.score(x_test,y_test)
mlp.score(x_test,y_test)