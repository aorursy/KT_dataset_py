# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reload_ext autoreload

%autoreload 2



from fastai import *

from fastai.tabular import *

import numpy as np



path = Path('/kaggle/input/house-prices-advanced-regression-techniques/')

path.ls()
train_df = pd.read_csv(path/'train.csv')

train_df.head()
n = len(train_df); n
idx = np.random.permutation(range(n))[:200]

idx.sort()



small_train_df = train_df.iloc[idx[:100]]

small_test_df = train_df.iloc[idx[100:]]

small_cont_vars = ['YearBuilt', '1stFlrSF', '2ndFlrSF']

small_cat_vars =  ['Neighborhood', 'Exterior1st']

small_train_df = small_train_df[small_cat_vars + small_cont_vars + ['SalePrice']]

small_test_df = small_test_df[small_cat_vars + small_cont_vars + ['SalePrice']]
small_train_df.head()
categorify = Categorify(small_cat_vars, small_cont_vars)

categorify(small_train_df)

categorify(small_test_df, test=True)
small_test_df.head()
fill_missing = FillMissing(small_cat_vars, small_cont_vars)

fill_missing(small_train_df)

fill_missing(small_test_df, test=True)
train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')
len_train, len_test = len(train_df),len(test_df)

print(len_train + len_test)
train_df['set'] = 'train'

test_df['set'] = 'test'
print(train_df['SalePrice'][:len_train])
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
procs=[FillMissing, Categorify, Normalize]
cat_vars = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1',

           'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

           'Foundation', 'Heating', 'Electrical', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 

           'SaleType', 'SaleCondition','CentralAir'] + ['ExterCond', 'ExterQual','BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',

           'BsmtFinType2','KitchenQual','Functional', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC','HeatingQC','LotShape',]



cont_vars = ['LotFrontage', 'LotArea',  'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',  '1stFlrSF', '2ndFlrSF', 

            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',  'Fireplaces', 

            'GarageYrBlt','MiscVal', 'MoSold', 'YrSold',

            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 

            ] 

            #+['Num_ExterCond', 'Num_ExterQual','Num_BsmtCond', 'Num_BsmtQual', 'Num_BsmtExposure', 'Num_BsmtFinType1','Num_BsmtFinType2',

            #'Num_KitchenQual','Num_Functional', 'Num_FireplaceQu', 'Num_GarageQual', 'Num_GarageCond', 'Num_PoolQC','Num_HeatingQC',

            #'Num_LotShape']
df = pd.concat([train_df,test_df])

df.head()

print(len(df))
fill_missing = FillMissing(cat_vars, cont_vars)

fill_missing(df)

fill_missing(df, test=True)



categorify = Categorify(cat_vars, cont_vars)

categorify(df)

categorify(df, test=True)



normalize = Normalize(cat_vars, cont_vars)

normalize(df)

normalize(df, test=True)
dep_var = 'SalePrice'

train_df = df.loc[df.set == 'train']

test_df = df.loc[df.set == 'test']
data = (TabularList.from_df(train_df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

                   #.split_by_rand_pct(valid_pct=0.3)

                   .split_none()

                   .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

                   .databunch()

       )
max_log_y = np.log(np.max(train_df['SalePrice'])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
import math

import torch



def rmsle(inp,targ):

    "Mean squared error between `inp` and `targ`."

    return torch.sqrt(msle(inp, targ))
learn = tabular_learner(data, layers=[1000,500], ps=[0.02,0.04], emb_drop=0.05, 

                        y_range=y_range, metrics=rmsle, loss_func=rmsle)

learn.model_dir = '/kaggle/working/'
learn.model
len(data.train_ds.cont_names)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(50, 1e-2, wd=0.5)
learn.save('lr 1e-1')
#learn.fit_one_cycle(15, 5e-3, wd=0.5)
learn.recorder.plot_losses()
#learn.save('lr 5e-4')
#learn.load('lr 5e-4')
test_df.head()
test_data = (TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=data.processor))

print(type(test_data[0]))

print(test_data[0]['Id'])
test_data = (TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=data.processor))

results = []

ids = []



for i in range(len(test_data)):

    print(i)

    ids.append(test_data[i]['Id'])

    #preds = learn.get_preds(test_data)

    pred = learn.predict(test_data[i])

    pred = torch.exp(pred[1])

    results.append(float(pred))

print(results)

submission = pd.DataFrame({'Id': ids, 'SalePrice':results})

print(submission)

submission.to_csv("submission.csv", index=False)
print(len(submission))

print(len(test_df))
#results = pd.DataFrame({'Id': test_df['Id'], 'SalePrice':list(preds)})

#results.to_csv("submission.csv")

#results.head()

#!kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "Message"
#test_data.to_df()