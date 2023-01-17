import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib as plt

#import plotly.graph_objs as go
#import plotly.plotly as py
#from plotly import tools
#from plotly.offline import iplot, init_notebook_mode
#init_notebook_mode()

import tensorflow as tf
#import tensorflow.contrib.eager as tfe # imperative 
#tf.enable_eager_execution()

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head(10)
train.info()
train.isna().sum().plot(kind='barh', figsize=(16,16));
cols =  train.columns
FEATURES = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
LABEL = 'SalePrice'
train[FEATURES].hist(figsize=(16,16));
train['GarageYrBlt'].ffill(inplace=True)
test['GarageYrBlt'].ffill(inplace=True)
for c in ['PoolQC','Fence', 'MiscFeature','Alley']:
    train[c].fillna(c[0],inplace=True)
    test[c].fillna(c[0],inplace=True)
for i,col in enumerate(test.columns):
    tr = train[col]
    ts = test[col]
    m = tr.value_counts().index[0]
    if tr.isna().sum()>0:
        #print('train',i,tr.dtype)
        tr.fillna(m,inplace=True)
    if ts.isna().sum()>0:
        #print('test',i,t.dtype,col,ts.isna().sum())  
        ts.fillna(m,inplace=True)
num_epochs =1000
#train.dtypes
num_cols = []
cat_cols = []
[num_cols.append(k) if train[k].dtype != object else cat_cols.append(k) for k in FEATURES]
#numeric_cols = list(filter(lambda x: x!='pass' , numeric_cols))
train[num_cols].head(10)
train[cat_cols].head(10)
def get_input_fn(data_set,shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
numeric_cols = [tf.feature_column.numeric_column(k) for k in num_cols]
#numeric_cols = list(filter(lambda x: x!='pass' , numeric_cols))
categorical = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(k,vocabulary_list=train[k].value_counts().index)) for k in cat_cols]
len(categorical)
model = tf.estimator.DNNRegressor(hidden_units=[2048,1024,512,1024,512,256,128],feature_columns=numeric_cols+categorical,
                                 optimizer=tf.train.AdagradOptimizer(
                                     learning_rate=0.0001))
model.train(input_fn= get_input_fn(train, shuffle =True),steps=4000)
result = model.evaluate(input_fn=get_input_fn(train[:128]))
for key in sorted(result):
    print(f'{key} -- {result[key]}')
test.head()
test_in = tf.estimator.inputs.pandas_input_fn(test[FEATURES], shuffle=False)
test_in
pred_iter = model.predict(input_fn=test_in)
predC = []
for i,pred in enumerate(pred_iter):
    #print(i,test['Id'][i],pred['predictions'][0])
    predC.append(pred['predictions'][0])  
len(predC)        
out_df = pd.DataFrame({"Id":test['Id'], "SalePrice":predC})
out_df.isna().sum()
file = out_df.to_csv("new.csv",index=False)
sample_submission = pd.read_csv('../working/new.csv')
#sample_submission.head(10)
sample_submission.isna().sum()