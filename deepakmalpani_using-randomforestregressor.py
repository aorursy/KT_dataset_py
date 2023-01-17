# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

test_id=test.ID

print(train.shape)

print(test.shape)

#train.head()

y=train.target

y=np.log(y+1)

#y.head()

train=train.drop(['ID'],axis=1)

train=train.drop(['target'],axis=1)

test=test.drop(['ID'],axis=1)
unique_df=train.nunique().reset_index()

unique_df.columns=["column","unique_count"]

constant_df=unique_df[unique_df["unique_count"]==1]

#constant_df.shape

train=train.drop(constant_df['column'],axis=1)

test=test.drop(constant_df['column'],axis=1)

print(train.shape)

print(test.shape)
total=(train==0).sum().sort_values(ascending=True)

percent=((train==0).sum()/((train==0).count())*100).sort_values(ascending=False)

train_data=pd.concat([total,percent],axis=1,keys=['total','percent'],sort=False)

#train_data.head(50)

cols_used=train_data[train_data.percent<79]

#cols_used.shape

cols_used=cols_used.index

#cols_used.shape

print(cols_used)

train=train[cols_used]

test=test[cols_used]

print(train.shape)

print(test.shape)
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

b=pca.fit_transform(train)

pca.transform(test)

#print(train.shape)

b
def rmsle(x,y):

    return np.sqrt(np.square(np.log(x+1)-np.log(y+1)).mean())
'''from sklearn.model_selection import GridSearchCV

model=RandomForestRegressor()

max_features=[0.5,0.75,0.85,0.95]

min_samples_leaf=np.arange(1,15)

min_samples_split=np.arange(2,15)

n_estimators=np.arange(1,11)*10

params={'max_features':max_features,'min_samples_leaf':min_samples_leaf,'min_samples_split':min_samples_split,'n_estimators':n_estimators}

grid_search=GridSearchCV(model,param_grid=params)

grid_search.fit(train,y)'''

model=RandomForestRegressor(max_features=0.75,n_estimators=100,min_samples_leaf=11,min_samples_split=13)

model.fit(train,y)

print(model.score(train,y))

y_pred=model.predict(X_test)

print(rmsle(np.exp(y_pred)-1,np.exp(y_test)-1))
out_test=model.predict(test)

out_test=np.exp(out_test)-1

out_df=pd.DataFrame(out_test)

out_df.columns = ['target']

out_df.insert(0, 'ID', test_id)

#out_df

out_df.to_csv('santander_submission.csv',index=False)
train=pd.read_csv('../input/train.csv')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

features=['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb']

y_train=train.target

X_train=train[features]
print(y_train.head())
print(X_train.head())
model=RandomForestRegressor(max_features=0.75,n_estimators=100,min_samples_leaf=11,min_samples_split=13)

model.fit(X_train,y_train)
model.score(X_train,y_train)
X_test=test.drop(['ID'],axis=1)

X_test.head()
X_test=X_test[features]

test_id=test.ID

out_test=model.predict(X_test)

out_test=np.exp(out_test)-1

out_df=pd.DataFrame(out_test)

out_df.columns = ['target']

out_df.insert(0, 'ID', test_id)

#out_df

out_df.to_csv('santander_submission2.csv',index=False)
