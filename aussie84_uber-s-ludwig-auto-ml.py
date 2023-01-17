!pip install https://github.com/uber/ludwig/archive/master.zip
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import ludwig

from ludwig.api import LudwigModel

import scipy as scipy



import matplotlib.pyplot as plt

%matplotlib inline
traindf = pd.read_csv("../input/train.csv")

testdf = pd.read_csv("../input/test.csv")
print(traindf.shape)

print(testdf.shape)
traindf.head()
testdf.head()
# Save the original for later

traindf_old = traindf 

testdf_old = testdf
all_data = pd.concat((traindf.loc[:,'MSSubClass':'SaleCondition'], testdf.loc[:,'MSSubClass':'SaleCondition']))
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = traindf[numeric_feats].apply(lambda x: scipy.stats.skew(x.dropna()))

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data = all_data.fillna(all_data.mean())
traindf = all_data.iloc[:len(traindf),:]

testdf = all_data.iloc[len(testdf)+1:,:]

traindf["SalePrice"] = np.log1p(traindf_old['SalePrice'])

print(traindf.shape)

print(testdf.shape)

print(traindf_old.shape)

print(testdf_old.shape)

# the ID columns is removed in the new traindf and testdf
traindf["SalePrice"].plot.hist()
traindf.head()
columns = list(traindf.columns)
dtypes = []

for c in columns:

    dtypes.append(traindf[c].dtype)
dtypes2 = []

for d in dtypes:

    if d in ('int64', 'float64'):

        dtypes2.append('numerical')

    if d == object:

        dtypes2.append('category')
print(dtypes2)
input_features = []

for col, dtype in zip(columns[:-1],dtypes2[:-1]):

    input_features.append(dict(name=col,type=dtype))

print(input_features)
model_definition = {

    'input_features':input_features,

    'output_features':[

        {'name': 'SalePrice', 'type':'numerical'}

    ]

}
model = LudwigModel(model_definition)

trainstats = model.train(traindf)
print(trainstats)
for i in trainstats.keys():

    print(i)

    for j in trainstats[i]:

        print(' ',j)

        for k in trainstats[i][j]:

            print('  ',k)

    print('--')
print(trainstats['train'].keys())
fig, axes = plt.subplots(1,2,figsize=(15,6))

axes[0].plot(trainstats['train']['SalePrice']['loss'],label='train')

axes[0].plot(trainstats['validation']['SalePrice']['loss'],label='validation')

axes[0].plot(trainstats['test']['SalePrice']['loss'],label='test')

axes[0].legend(loc='upper right')

axes[0].set_title('Loss')

axes[1].plot(trainstats['train']['SalePrice']['mean_absolute_error'],label='train')

axes[1].plot(trainstats['validation']['SalePrice']['mean_absolute_error'],label='validation')

axes[1].plot(trainstats['test']['SalePrice']['mean_absolute_error'],label='test')

axes[1].legend(loc='upper right')

axes[1].set_title('mean_absolute_error')

plt.show()
pd.set_option('display.max_columns', 10)

print(traindf.head())

print(testdf.head())

print(traindf_old.head())

print(testdf_old.head())
testdf.head()
predictions = model.predict(testdf)
predictions.plot.hist()
predictions.head(10)
predictions = np.expm1(predictions)
pd.options.display.float_format = '{:,.0f}'.format

predictions.head(10)
submission = pd.DataFrame(testdf_old['Id'])

submission['SalePrice'] = predictions

submission.head()
submission.to_csv('submission.csv',index=False)