

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import tensorflow as tf

import tensorflow_estimator.python.estimator.api._v2.estimator

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = train[train['GrLivArea']<4000]

train["SalePrice"] = np.log1p(train["SalePrice"])

print(train.head())



corr=train.corr()

plt.figure(figsize=[30,15])

sns.heatmap(corr, annot=True)
train.PoolQC=train.PoolQC.fillna('NoPool')

train.MiscFeature=train.MiscFeature.fillna('NoFeature')

train.Fence=train.Fence.fillna('NoFence')

train.Alley=train.Alley.fillna('NoAlley')

train.FireplaceQu=train.FireplaceQu.fillna('NoFire')

train.BsmtQual=train.BsmtQual.fillna('NoBasement')

train.MasVnrArea=train.MasVnrArea.fillna(0)

train.GarageType=train.GarageType.fillna('NoGarage')

train.GarageFinish=train.GarageFinish.fillna('NoGarage')

train.GarageQual=train.GarageQual.fillna('NoGarage')

#train.drop('LotFrontage',axis=1,inplace=True)

train.drop('GarageYrBlt',axis=1,inplace=True)

train.drop('GarageCars',axis=1,inplace=True)

#train.drop('GarageCars',axis=1,inplace=True)

train.drop('GarageQual',axis=1,inplace=True)

train.drop('BsmtCond',axis=1,inplace=True)

train.drop('BsmtFinSF1',axis=1,inplace=True)

#train.drop('LotFrontage',axis=1,inplace=True)

#train.drop('GarageYrBlt',axis=1,inplace=True)

train.drop('GarageCond',axis=1,inplace=True)

train.drop('Electrical',axis=1,inplace=True)

train.drop('BsmtFinType2',axis=1,inplace=True)



ina=train.isna().sum().sort_values(ascending=False)

print(ina[ina>0])

#print(train.head(100))



print(test.isna().sum())



test.PoolQC=test.PoolQC.fillna('NoPool')

test.MiscFeature=test.MiscFeature.fillna('NoFeature')

test.Fence=test.Fence.fillna('NoFence')

test.Alley=test.Alley.fillna('NoAlley')

test.FireplaceQu=test.FireplaceQu.fillna('NoFire')

test.BsmtQual=test.BsmtQual.fillna('NoBasement')

test.MasVnrArea=test.MasVnrArea.fillna(0)

test.GarageType=test.GarageType.fillna('NoGarage')

test.GarageFinish=test.GarageFinish.fillna('NoGarage')

test.GarageQual=test.GarageQual.fillna('NoGarage')

#test.drop('LotFrontage',axis=1,inplace=True)

test.drop('GarageYrBlt',axis=1,inplace=True)

test.drop('GarageCars',axis=1,inplace=True)

#test.drop('GarageCars',axis=1,inplace=True)

test.drop('GarageQual',axis=1,inplace=True)

test.drop('BsmtCond',axis=1,inplace=True)

test.drop('BsmtFinSF1',axis=1,inplace=True)

#test.drop('LotFrontage',axis=1,inplace=True)

#test.drop('GarageYrBlt',axis=1,inplace=True)

test.drop('GarageCond',axis=1,inplace=True)

test.drop('Electrical',axis=1,inplace=True)

test.drop('BsmtFinType2',axis=1,inplace=True)







categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

print(numerical_features)

numerical_features = numerical_features.drop("SalePrice")

print(categorical_features)

#numerical_features.drop('SalePrice',inplace=True)

train_num=train[numerical_features]

train_cat=train[categorical_features]



categorical_featurest = test.select_dtypes(include = ["object"]).columns

numerical_featurest = test.select_dtypes(exclude = ["object"]).columns

print(numerical_featurest)

print(categorical_featurest)

#numerical_featurest.drop('SalePrice',inplace=True)

test_num=test[numerical_featurest]

test_cat=test[categorical_featurest]

test_num=test_num.fillna(test_num.median())



print("Length of numeric features:",len(numerical_features))

print("Length of categorical features:",len(categorical_features))

ina=train.isna().sum().sort_values(ascending=False)

print(ina[ina>0])

#labels=train_num.drop['SalePrice']

print("NA Values:",train_num.isnull().values.sum())

train_num=train_num.fillna(train_num.median())

print("NA Values:",train_num.isnull().sum())
print(train_cat.shape)

print(test_cat.shape)

train_cat=pd.get_dummies(train_cat)

test_cat=pd.get_dummies(test_cat)

missing=set(train_cat)-set(test_cat)

for col in missing:

    test_cat[col]=0

print(train_cat.shape)

print(test_cat.shape)

print(train_cat.isnull().values.sum())

trainf=pd.concat([train_num,train_cat],axis=1)

testf=pd.concat([test_num,test_cat],axis=1)

trainf.drop('Id',axis=1,inplace=True)

Id=testf.pop('Id')

print(Id)

print(trainf.head())

print(testf.head())

print(test.isnull().sum())
X=tf.constant(trainf.values)

y=tf.constant(train.SalePrice.values)

testx=X[600:]

testy=y[600:]



from tensorflow.keras.layers import Dense

model=tf.keras.Sequential()

model.compile(optimizer='adam',loss='mse')

model.add(Dense(4))

model.add(Dense(12,kernel_regularizer=tf.keras.regularizers.l2(l=0.1)))

model.add(Dense(12,kernel_regularizer=tf.keras.regularizers.l2(l=0.1)))

model.add(Dense(12,kernel_regularizer=tf.keras.regularizers.l2(l=0.1)))

model.add(Dense(1))

model.fit(X,y,epochs=300)

a=model.predict(testf)
print(a)

print(a.shape)

print(Id)

a=pd.DataFrame(a)

a= np.expm1(a)

print(a)

id=pd.DataFrame(Id)

id['SalePrice']=a

print(id)

aa=pd.concat([id,a],axis=1)

aa[0].column='SalePrice'

print(Id.shape)

print(aa)

#sub=pd.DataFrame({'Id':Id,'SalePrice':a})

id.to_csv('submission.csv',index=False)