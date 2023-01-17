# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv("/kaggle/input/train.csv")
df1.head()
df1['Month'] = df1['week'].apply(lambda x: int(x / 4))

df1['Year'] = df1['week'].apply(lambda x: int(x / 52))

df1['Quarter'] = df1['week'].apply(lambda x: int(x / 13))

df2=pd.read_csv("/kaggle/input/meal_info.csv")
df2.head()
df3=pd.read_csv("/kaggle/input/fulfilment_center_info.csv")
df3.head()
df4=pd.merge(df1, df2, on='meal_id')

train=pd.merge(df4, df3, on='center_id')
train.head()
train.info()
train.loc[train['checkout_price'] < train['base_price'], 'C'] = 1

train.loc[train['checkout_price'] > train['base_price'], 'C'] = 2

train.loc[train['checkout_price'] == train['base_price'], 'C'] = 0
train.head()
train["category"].value_counts()
train["cuisine"].value_counts()
train["center_type"].value_counts()
train.shape
train.head()
import seaborn as sns

plt.figure(figsize=(20,20))

sns.heatmap(train.corr(),annot=True)
del train["id"]
train.info()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()



train["category"]=le.fit_transform(train["category"])

train["cuisine"]=le.fit_transform(train["cuisine"])

train["center_type"]=le.fit_transform(train["center_type"])
train["center_id"].value_counts()
train["meal_id"].value_counts()
train["center_id"]=le.fit_transform(train["center_id"])

train["meal_id"]=le.fit_transform(train["meal_id"])

train["city_code"]=le.fit_transform(train["city_code"])

train["region_code"]=le.fit_transform(train["region_code"])
train.head()
train["ratio"]=train["checkout_price"]/train["base_price"]

train['ratio'] =train['ratio'].apply(lambda x: 1 if(x<0.5) else x)
train.loc[train['checkout_price'] <= 100, 'Cat'] = 0

train.loc[(train['checkout_price'] > 100) & (train['checkout_price'] <= 150), 'Cat'] = 1

train.loc[(train['checkout_price'] > 150) & (train['checkout_price'] <= 200), 'Cat'] = 2

train.loc[(train['checkout_price'] > 200) & (train['checkout_price'] <= 300), 'Cat'] = 3

train.loc[(train['checkout_price'] > 300) & (train['checkout_price'] <= 500), 'Cat'] = 4

train.loc[(train['checkout_price'] > 500) , 'Cat'] = 5
#df_last=train.loc[train.groupby(['center_id','meal_id']).week.idxmax()]

#df_last=df_last[['meal_id','center_id','num_orders']]

#df_last=df_last.rename(columns={'num_orders':'last_order'})

#df_last=df_last.reset_index()
#df_last.head()
df_last3=train[(train['week']==143) | (train['week']==144) | (train['week']==145) ]

df_last3_group=df_last3.groupby(['center_id','meal_id'])["num_orders"].mean()

df_group_3=df_last3_group.reset_index()

df_group_3=df_group_3.rename(columns={'num_orders':'avg_3_orders'})
df_group_3.head()
df_group=train.groupby(['center_id','meal_id'])["num_orders"].mean()

df_group=df_group.reset_index()

df_group=df_group.rename(columns={'num_orders':'avg_orders'})
df_group.head()
df_new = pd.merge(train, df_group,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])

df_new = pd.merge(df_new, df_group_3,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
df_new.head()
test1=pd.read_csv("/kaggle/input/test.csv")
test1['Month'] = test1['week'].apply(lambda x: int(x / 4))

test1['Year'] = test1['week'].apply(lambda x: int(x / 52))

test1['Quarter'] = test1['week'].apply(lambda x: int(x / 13))

test1.head()
df_3=pd.merge(test1, df2, on='meal_id')

test=pd.merge(df_3, df3, on='center_id')
test.loc[test['checkout_price'] < test['base_price'], 'C'] = 1

test.loc[test['checkout_price'] > test['base_price'], 'C'] = 2

test.loc[test['checkout_price'] == test['base_price'], 'C'] = 0

test["category"]=le.fit_transform(test["category"])

test["cuisine"]=le.fit_transform(test["cuisine"])

test["center_type"]=le.fit_transform(test["center_type"])

test["center_id"]=le.fit_transform(test["center_id"])

test["meal_id"]=le.fit_transform(test["meal_id"])

test["city_code"]=le.fit_transform(test["city_code"])

test["region_code"]=le.fit_transform(test["region_code"])

test["ratio"]=test["checkout_price"]/test["base_price"]

test['ratio'] =test['ratio'].apply(lambda x: 1 if(x<0.5) else x)
test.loc[test['checkout_price'] <= 100, 'Cat'] = 0

test.loc[(test['checkout_price'] > 100) & (test['checkout_price'] <= 150), 'Cat'] = 1

test.loc[(test['checkout_price'] > 150) & (test['checkout_price'] <= 200), 'Cat'] = 2

test.loc[(test['checkout_price'] > 200) & (test['checkout_price'] <= 300), 'Cat'] = 3

test.loc[(test['checkout_price'] > 300) & (test['checkout_price'] <= 500), 'Cat'] = 4

test.loc[(test['checkout_price'] > 500) , 'Cat'] = 5
del test["id"]
test.head()
df_new_2 = pd.merge(test, df_group,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])

df_new_2 = pd.merge(df_new_2, df_group_3,  how='left', left_on=['center_id','meal_id'], right_on = ['center_id','meal_id'])
train=df_new

test=df_new_2
print(train.shape , test.shape)
y=train["num_orders"]

del train["num_orders"]

x=train
x.isnull().sum()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
params = {}

params["objective"] = "reg:linear"

params["eta"] = 0.03

params["min_child_weight"] = 10

params["subsample"] = 0.8

params["colsample_bytree"] = 0.7

params["silent"] = 1

params["max_depth"] = 10

#params["max_delta_step"]=2

params["seed"] = 0

 #params['eval_metric'] = "auc"

plst4 = list(params.items())

num_rounds4 = 1100



import xgboost as xgb

xgdmat=xgb.DMatrix(x_train,y_train)



final_gb4=xgb.train(plst4,xgdmat,num_rounds4)



tesdmat=xgb.DMatrix(x_test)

y_pred=final_gb4.predict(tesdmat)



from sklearn.metrics import mean_squared_error as mse

print(np.sqrt(mse(y_pred,y_test)))
testdmat=xgb.DMatrix(test)

pred1=final_gb4.predict(testdmat)
pred1
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,max_features = "auto", min_samples_leaf = 10)
model.fit(x,y)
pred2=model.predict(test)
pred2
sam=pd.read_csv("sample.csv")
sam["num_orders"]=pred2
j=0

for i in sam["num_orders"]:

  if(i<=0):

    sam["num_orders"][j]=13

  j+=1
sam.to_csv("s2.csv",index=False)