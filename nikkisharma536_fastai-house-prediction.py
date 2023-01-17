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
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *

from fastai.tabular import *
# Read and load Data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")


f = open('../input/data_description.txt', 'r') 

print (f.read())

f.close()
train.head()
# check index of dataframe

train.columns
dep_var = 'SalePrice'

# cat_names = train.select_dtypes(include=['object']).columns.tolist()

cat_names = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 

             'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 

             'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 

             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

             'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

             'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',

             'MSSubClass', 'OverallQual', 'OverallCond','BsmtFullBath',

              'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

              'Fireplaces','GarageCars','YrSold' , 'MoSold', 'LowQualFinSF' , 'PoolArea', 

             'YearBuilt', 'YearRemodAdd' , 'MiscVal', '3SsnPorch']



# cont_names = train.select_dtypes(include=[np.number]).columns.tolist()

cont_names = [  'LotFrontage', 'LotArea',  

               'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 

              'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea',

              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']



print("Categorical columns are : ", cat_names)

print('Continuous numerical columns are :', cont_names)

procs = [FillMissing, Categorify, Normalize]
test_id = test['Id']

test.fillna(value = test.mean(), inplace = True)
test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .split_by_rand_pct(valid_pct = 0.3, seed = 42)

                        .label_from_df(cols = dep_var, label_cls = FloatList, log = True )

                        .add_test(test)

                        .databunch())

data.show_batch(rows=10)

learn = tabular_learner(data, layers=[200,100], metrics=rmse)

#learn = tabular_learner(data,

                            #layers=[100,50,1])

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(15, max_lr =1e-01)

# get predictions

preds, targets = learn.get_preds(DatasetType.Test)
labels = [np.exp(p[0].data.item()) for p in preds]
submission = pd.DataFrame({'Id': test_id, 'SalePrice': labels})

submission.to_csv('submission.csv', index=False)

submission.head()