from fastai import *

from fastai.tabular import *

import numpy as np

path= Path("../input"); path.ls()
import pandas as pd

train_df = pd.read_csv(path/'train.csv')

train_df.head(5)
train_df.columns
train_df.index = train_df['Id']
del train_df['Id']
train_df.columns
dep_var = 'SalePrice'

cat_names = [

    'MSZoning',

    'LotFrontage',

    'Street',

    'Alley',

    'LotShape',

    'LandContour',

    'Utilities',

    'LotConfig',

    'LandSlope',

    'Neighborhood',

    'Condition1',

    'Condition2',

    'BldgType',

     'HouseStyle',

     'RoofStyle',

     'RoofMatl',

     'Exterior1st',

     'Exterior2nd',

     'MasVnrType',

     'MasVnrArea',

     'ExterQual',

     'ExterCond',

     'Foundation',

     'BsmtQual',

     'BsmtCond',

     'BsmtExposure',

     'BsmtFinType1',

     'BsmtFinType2',

     'Heating',

     'HeatingQC',

     'CentralAir',

     'Electrical',

     'KitchenQual',

     'Functional',

     'FireplaceQu',

     'GarageType',

     'GarageYrBlt',

     'GarageFinish',

     'GarageQual',

     'GarageCond',

     'PavedDrive',

     'PoolQC',

     'Fence',

     'MiscFeature',

     'SaleType',

     'SaleCondition',

     'SalePrice',

]

cont_names = [

    'MSSubClass',

    'LotArea',

    'OverallQual',

    'OverallCond',

    'YearBuilt',

    'YearRemodAdd',

    'BsmtFinSF1',

    'BsmtFinSF2',

    'BsmtUnfSF',

    'TotalBsmtSF',

    '1stFlrSF',

    '2ndFlrSF',

    'LowQualFinSF',

    'GrLivArea',

    'BsmtFullBath',

    'BsmtHalfBath',

    'FullBath',

    'HalfBath',

    'BedroomAbvGr',

    'KitchenAbvGr',

    'TotRmsAbvGrd',

    'Fireplaces',

    'GarageCars',

    'GarageArea',

    'WoodDeckSF',

    'OpenPorchSF',

    'EnclosedPorch',

    '3SsnPorch',

    'ScreenPorch',

    'PoolArea',

    'MiscVal',

    'MoSold',

    'YrSold',

]
procs = [FillMissing, Categorify, Normalize]
test_df = pd.read_csv(path/'test.csv')

test_df.index = test_df['Id']



for i in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',]: #cont vars which are not nan in training set

    test_df[i]= test_df[i].fillna(test_df[i].mean())



for i in ['GarageCars','GarageArea']:

    test_df[i]= test_df[i].fillna(0)



#https://forums.fast.ai/t/should-the-dep-var-be-in-test-df-for-tabular-data-from-df/25373/2

test_df[dep_var] = 0.0



del test_df['Id']

test_df.head()
test = TabularList.from_df(df=test_df, cat_names=cat_names,cont_names=cont_names)
#train_df[dep_var] = np.log(train_df[dep_var])
data = (

        TabularList

        .from_df(df=train_df,cat_names=cat_names,cont_names=cont_names, procs = procs)

        .split_by_rand_pct(0.2, seed=42)

        .label_from_df(cols=dep_var,label_cls=FloatList)

        .add_test(test)

        .databunch()

       )
data.show_batch(rows=10)
max_log_y = np.max(train_df[dep_var])*1.2



y_range = torch.tensor([0,max_log_y],device=defaults.device)

max_log_y,y_range
learn = tabular_learner(

    data,

    layers=[1000 ,500],

    metrics=rmse,

    emb_drop=0.04,

    ps=[0.001,0.01]

)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 1e-1)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(500, 0.01)
learn.lr_find()

learn.recorder.plot()
preds, _ = learn.get_preds(DatasetType.Test)
preds
submission_df = pd.read_csv(path/'test.csv',usecols=['Id'])
submission_df['SalePrice'] = preds.numpy()
submission_df.to_csv('submission.csv',index=False)
submission_df.head()