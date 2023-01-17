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
from fastai.tabular import *
df = pd.read_csv('../input/train.csv')

df.head()
test_df = pd.read_csv('../input/test.csv')
df.columns
len(df.columns)
df['BsmtHalfBath'].unique()
dep_var = 'SalePrice'

cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',

           'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 

           'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',

           'Electrical', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

           'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'BsmtQual', 'KitchenQual']

cont_vars = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr',

 

 'EnclosedPorch', 'Fireplaces', 'FullBath',

 'GarageYrBlt', 'GrLivArea',

 'HalfBath', 'KitchenAbvGr', 

 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea',

 'OpenPorchSF', 'PoolArea', 'ScreenPorch',

 'TotRmsAbvGrd', 'WoodDeckSF']
procs = [FillMissing, Categorify, Normalize]
data = (TabularList.from_df(df, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

       .split_by_rand_pct()

       .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

        .add_test(TabularList.from_df(test_df, cat_names=cat_vars, cont_names=cont_vars))

       .databunch())
max_log_y = np.log(np.max(df[dep_var])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
learn = tabular_learner(data, layers=[1000,500], y_range=y_range, metrics=exp_rmspe)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=1e-2)
learn.save('stage-1')
learn.load('stage-1');
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=5e-3)
learn.save('stage-2')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 3e-4)
learn.save('stage-3')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 1e-3)
learn.save('stage-4')
def create_submission(learn:Learner, name='model'):

    name = name + '_submission.csv'

    

    test_data = pd.read_csv('../input/test.csv')

    result = pd.DataFrame(columns=['Id', 'SalePrice'])

    preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    result['SalePrice'] = np.exp(preds.data).numpy().T[0]

    result['Id'] = test_data['Id'].T

    return result

    
sub = create_submission(learn)
sub.head()
sub.index += 1
sub.head()
results.to_csv('my_submission.csv')