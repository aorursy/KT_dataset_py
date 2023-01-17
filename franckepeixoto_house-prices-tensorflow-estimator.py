from IPython.display import clear_output

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

import numpy as np

import os

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("./"))
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#check ids.

print('test  min\t',test.Id.min())

print('test  max\t',test.Id.max())

print('train min\t',train.Id.min())

print('train max\t',train.Id.max())

print('--')

print('min sale price\t',train.SalePrice.max())

print('max sale price\t',train.SalePrice.min())

print('count sale price',train.SalePrice.count())



plt.figure(figsize=(10,1))

sns.distplot(train.SalePrice)

plt.legend(['Sale Price'])

plt.axis('off')
plt.figure(figsize=(10,3))

sns.scatterplot(  x="GrLivArea", y="SalePrice",data=train)

plt.legend(['GrLivArea x Sale Price'])

plt.axis('on')
train = train[train.GrLivArea < 4000]

plt.figure(figsize=(10,3))

sns.scatterplot(  x="GrLivArea", y="SalePrice",data=train)

plt.legend(['GrLivArea x Sale Price'])

plt.axis('on')
train_id = train.Id

test_id = test.Id

data = train.copy()

train = train[['Id','SalePrice']]

data.drop("SalePrice", axis = 1, inplace = True)

data =  pd.concat([data,test],axis=0,sort=False)

data = data.reset_index(drop=True)

data.describe()
data.MSSubClass = data.MSSubClass.astype(str)

msSubClass= np.unique(data.MSSubClass.values)

for sub_class in msSubClass:

  if "SC" not in sub_class:

    data.loc[data["MSSubClass"] == sub_class,"MSSubClass"] = "SC"+sub_class

data.OverallQual = data.OverallQual.astype(str)

overallQual= np.unique(data.OverallQual.values)

for overall_qual in overallQual:

  if "OQ" not in overall_qual:

    data.loc[data["OverallQual"] == overall_qual,"OverallQual"] = "OQ"+overall_qual



#OverallCond : classificação geral das condições

data.OverallCond = data.OverallCond.astype(str)

overallCond= np.unique(data.OverallCond.values)

for overall_cond in overallCond:

  if "OC" not in overall_cond:

    data.loc[data["OverallCond"] == overall_cond,"OverallCond"] = "OC"+overall_cond



print('MSSubClass:',np.unique(data.MSSubClass.values))

print('--')

print('OverallQual:',np.unique(data.OverallQual.values))

print('--')

print('OverallCond:',np.unique(data.OverallCond.values))

data = data.reset_index(drop=True)
print("dtypes:",data.dtypes.unique())

quantitative_columns = [f for f in data.columns if data.dtypes[f] != 'object']

qualitative_columns = [f for f in data.columns if data.dtypes[f] == 'object']

quantitative_columns.pop(0)

print('qualitative columns:',qualitative_columns)

print('quantitative columns:',quantitative_columns)
total=data.isnull().sum().sort_values(ascending=False)

percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing=pd.concat([total,percent], axis=1,keys=['Total','%'])

missing.head(30)
data.Alley.mode()

data.Alley.fillna('NA', inplace=True) #No alley access

plt.figure(figsize=(4,2))

sns.barplot(x=data.Alley, y=train.SalePrice)

plt.axis('on')
data.LotFrontage.mode()

data.LotFrontage.fillna(data.LotFrontage.median(), inplace=True)

data.LotFrontage

plt.figure(figsize=(10,3))

sns.distplot(data.LotFrontage, hist_kws={'alpha':0.5}, label='LotFrontage')

plt.legend()
data.MasVnrType.mode()

data.MasVnrType.fillna('NA', inplace=True)

plt.figure(figsize=(8,2))

sns.barplot(x=data.MasVnrType, y=train.SalePrice)

plt.axis('on')
data.MasVnrArea.mode()

data.MasVnrArea.fillna(0.0, inplace=True)

plt.figure(figsize=(10,3))

sns.distplot(data.MasVnrArea, hist_kws={'alpha':0.4}, label='MasVnrArea')

plt.legend()
data.BsmtQual.mode()

data.BsmtQual.fillna('NA', inplace=True)

plt.figure(figsize=(12,2))

sns.barplot(x=data.BsmtQual, y=train.SalePrice)
data.BsmtCond.mode()

data.BsmtCond.fillna('NA', inplace=True)

plt.figure(figsize=(12,2))

sns.barplot(x=data.BsmtCond, y=train.SalePrice)
data.BsmtExposure.mode()

data.BsmtExposure.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.BsmtExposure, y=train.SalePrice)
data.BsmtFinType1.mode()

data.BsmtFinType1.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.BsmtFinType1, y=train.SalePrice)
data.BsmtFinSF1.mode()

data.BsmtFinSF1.fillna(0, inplace=True)

plt.figure(figsize=(10,3))

sns.distplot(data.BsmtFinSF1, hist_kws={'alpha':0.5}, label='BsmtFinSF1')

plt.legend()
data.BsmtFinType2.mode()

data.BsmtFinType2.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.BsmtFinType2, y=train.SalePrice)
data.Electrical.mode()

data.Electrical.fillna('SBrkr', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.Electrical, y=train.SalePrice)
data.FireplaceQu.mode()

data.FireplaceQu.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.FireplaceQu, y=train.SalePrice)
data.GarageType.mode()

data.GarageType.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.GarageType, y=train.SalePrice)
data.GarageYrBlt = data.GarageYrBlt.fillna(data.YearBuilt)#, inplace=True)

plt.figure(figsize=(10,2))

sns.scatterplot(x=data.GarageYrBlt, y=train.SalePrice)
data.GarageFinish.mode()

data.GarageFinish.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.GarageFinish, y=train.SalePrice)
data.GarageQual.mode()

data.GarageQual.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.GarageQual, y=train.SalePrice)
data.GarageCond.mode()

data.GarageCond.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.GarageCond, y=train.SalePrice)
data.PoolQC.mode()

data.PoolQC.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.PoolQC, y=train.SalePrice)
data.Fence.mode()

data.Fence.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.Fence, y=train.SalePrice)
data.MiscFeature.mode()

data.MiscFeature.fillna('NA', inplace=True)

plt.figure(figsize=(10,2))

sns.barplot(x=data.MiscFeature, y=train.SalePrice)
for col in  quantitative_columns:

  data[col].mode()

  data[col].fillna(0, inplace=True)
data_corr = data[:train.shape[0]].copy() 

data_corr = data_corr[quantitative_columns]

data_corr['SalePrice']  = train.SalePrice.values

data_corr = data_corr.reset_index(drop=True)

plt.figure(figsize=(20,10))

corr =data_corr.corr(method='pearson')

corr = corr[corr>=.4]

sns.heatmap(corr,annot=True,cmap='YlGnBu',fmt='.1f',linewidths=1)
best_corr = corr['SalePrice'].sort_values(ascending=False).to_dict()

best_columns=['GrLivArea',

 'GarageCars',

 'TotalBsmtSF',

 'GarageArea',

 '1stFlrSF',

 'FullBath',

 'TotRmsAbvGrd',

 'YearBuilt',

 'YearRemodAdd',

 'GarageYrBlt',

 'MasVnrArea',

 'Fireplaces',

 'BsmtFinSF1',

 'LotFrontage',

 'OpenPorchSF',

 'WoodDeckSF']

for key,value in best_corr.items():

    if ((value>=0.3175) & (value<0.9)) | (value<=-0.315):

        best_columns.append(key)

best_columns 
total=data.isnull().sum().sort_values(ascending=False)

percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing=pd.concat([total,percent], axis=1,keys=['Total','%'])



missing[missing['%']>0].head(30)
data.SaleType.mode()

data.SaleType.fillna('Oth', inplace=True)	

plt.figure(figsize=(12,2))

sns.barplot(x=data.SaleType, y=train.SalePrice)
data.KitchenQual.mode()

data.KitchenQual.fillna('TA', inplace=True)

plt.figure(figsize=(5,2))

sns.barplot(x=data.KitchenQual, y=train.SalePrice)
data.Exterior1st.mode()

data.Exterior1st.fillna('VinylSd', inplace=True)

plt.figure(figsize=(20,2))

sns.barplot(x=data.Exterior1st, y=train.SalePrice)
data.Exterior2nd.mode()

data.Exterior2nd.fillna('VinylSd', inplace=True)	

plt.figure(figsize=(20,2))

sns.barplot(x=data.Exterior2nd, y=train.SalePrice)
data.Utilities.mode()

data.Utilities.fillna('AllPub', inplace=True)	

plt.figure(figsize=(3,2))

sns.barplot(x=data.Utilities, y=train.SalePrice)
data.Functional.mode()

data.Functional.fillna('Typ', inplace=True)	

plt.figure(figsize=(8,2))

sns.barplot(x=data.Functional, y=train.SalePrice)
data.MSZoning.mode()

data.MSZoning.fillna('RL', inplace=True)	

plt.figure(figsize=(10,2))

sns.barplot(x=data.MSZoning, y=train.SalePrice)
total=data.isnull().sum().sort_values(ascending=False)

percent=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing=pd.concat([total,percent], axis=1,keys=['Total','%'])

missing[missing['%']>0].head(30)
data.describe().transpose()
size = train.shape[0]

orig_label = train.SalePrice.copy()

label = train.SalePrice.values
train = data[:size]

train['SalePrice']  = orig_label.values

label = train.SalePrice.values

train.drop("SalePrice", axis = 1, inplace = True)

test = data[size:]

train = train.reset_index(drop=True)

test = test.reset_index(drop=True)

train.head(3)
test.shape,train.shape
feature_columns = []

for column_name in  np.unique(best_columns):#quantitative_columns:

  feature_columns.append(tf.feature_column.numeric_column(column_name))



def one_hot_cat_column(feature_name, vocab):

  return tf.feature_column.indicator_column( tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))



for column_name in qualitative_columns:

  #vocabulary =np.unique(train[qualitative_columns[0]].values)

  vocabulary = train[column_name].unique()

  categorical_column =one_hot_cat_column(column_name,vocabulary)

  feature_columns.append(categorical_column)



print('feature_columns:\t',len(feature_columns))

print('feature_columns:\t',feature_columns)

train[best_columns].head(1)
train.interpolate(method='linear',inplace=True)

test.interpolate(method='linear',inplace=True)
batch_size = 1



#boost_testimator  = tf.estimator.BoostedTreesRegressor(feature_columns=feature_columns,max_depth=10, learning_rate=0.1, l1_regularization=0.1, l2_regularization=0.1, n_batches_per_layer=1,n_trees=700)

boost_testimator  = tf.estimator.BoostedTreesRegressor(feature_columns=feature_columns,max_depth=10, learning_rate=0.1,n_batches_per_layer=1,n_trees=3000)



epochs = 1

def input_estimator(xdata,ydata,epochs=None,shuffle=True):

  def input_fn():

    dataset = tf.data.Dataset.from_tensor_slices((dict(xdata), ydata))

    if shuffle:

        dataset = dataset.shuffle(len(ydata))

    dataset = dataset.repeat(epochs)

    dataset = dataset.batch(len(ydata))

    return dataset

  return input_fn

boost_testimator.train(input_estimator(train,label),max_steps=30)

clear_output()
results = boost_testimator.evaluate(input_estimator(train,label,epochs=1,shuffle=False))

clear_output()

pd.Series(results).to_frame()

print(pd.Series(results))
predict_input_fn = lambda: tf.data.Dataset.from_tensors(dict(test))

preds = np.array([p['predictions'][0] for p in boost_testimator.predict(predict_input_fn)])
preds.shape,test.shape
submission = pd.DataFrame({"ID" : test_id, "SalePrice" : preds})

submission.to_csv("prediction_values.csv", index=False)

submission.head(1)