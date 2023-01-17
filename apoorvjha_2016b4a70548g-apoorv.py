import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as skl

import math

%matplotlib inline



np.random.seed(0)
train = pd.read_csv('../input/bits-f464-l1/train.csv', sep=',')

train.head()
test = pd.read_csv('../input/bits-f464-l1/test.csv', sep=',')
id_train = train['id']

train=train.drop(['id'],axis=1)

id_test=test['id']

test=test.drop(['id'],axis=1)
x_train = train.drop(["label"], axis = 1)

y_train = train.loc[ : , "label"]
x_train.shape
corr_matrix = train.corr().abs()

pd.options.display.max_rows = None

corr_matrix["label"].sort_values()
x_train['b10']
x_train = x_train.drop(["b10",'b12','b26','b61','b81'], axis = 1)

test = test.drop(["b10",'b12','b26','b61','b81'], axis = 1)
corr_matrix = x_train.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features 

x_train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)
from scipy.stats import skew

numeric_feats = x_train.dtypes[x_train.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = x_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

print(skewness.head(30))
x_train = x_train.drop(["b58",'b68','b72','b64','b21','b86','b82'], axis = 1)

test = test.drop(["b58",'b68','b72','b64','b21','b86','b82'], axis = 1)
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(x_train,y_train,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train, y_train)

pred=regressor.predict(test)
res1 = pd.DataFrame(pred)

res1.insert(0,"id",id_test,True)

res1 = res1.rename(columns={0: "label"})

print(res1.head())

res1.to_csv('done.csv', index = False)