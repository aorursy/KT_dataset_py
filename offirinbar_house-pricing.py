# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in AttributeError: Caught AttributeError in DataLoader worker process 0.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import fast.ai library



from fastai import *

from fastai.tabular import *

from fastai.vision import *



#import more libraries

import seaborn as sns

import pandas_profiling
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_train.head()
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

#df_test.head()
combine = [df_train, df_test]
#df_train.describe()

df_train.info()

print('_'*40)

df_test.info()
df_train.describe()
df_train.describe(include=['O'])
#Explore the data:



#pivot_ui(df_train) - > usued in the local computer to explore the data.

#pandas_profiling.ProfileReport(df_train)

#df_train[df_train.columns[1:]].corr()['SalePrice'][:].sort_values(ascending=False)
'''

#deleting columns



print("Before", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)



df_train = df_train.drop(['Alley','Id','Utilities','FireplaceQu','PoolQC','Fence','MiscFeature'], axis =1)

df_test = df_test.drop(['Alley','Id','Utilities','FireplaceQu','PoolQC','Fence','MiscFeature'], axis =1)

combine = [df_train, df_test]



print("After", df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)



'''
dep_var = 'SalePrice'

cat_vars = ['MSSubClass','MSZoning','LotFrontage','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF2','Heating','HeatingQC','CentralAir','Electrical','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageQual','GarageCond','PavedDrive','3SsnPorch','PoolArea','MiscVal','MoSold','YrSold','SaleType','SaleCondition',]

cont_vars = ['LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch',]

procs = [FillMissing, Categorify, Normalize]
#defining the range of the target (same idea as rossman)



#max_log_y = np.log(np.max(df_train[dep_var])*1.2)

#y_range = torch.tensor([0, max_log_y], device=defaults.device)



#printing it 



#max_log_y,y_range



test_id = df_test['Id']



df_test.fillna(value = df_test.mean(), inplace = True)

test = TabularList.from_df(df_test, cat_names=cat_vars, cont_names=cont_vars)
#we use: label_cls = FloatList because its a regrresion problem. log=true for rmse

np.random.seed(42)

data = (TabularList.from_df(df_train, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

                        .split_by_rand_pct(valid_pct = 0.2)

                        .label_from_df(cols = dep_var, label_cls = FloatList, log = True )

                        .add_test(test)

                        .databunch())
max_log_y = np.log(np.max(df_train[dep_var])*1.25)

y_range = torch.tensor([0, max_log_y], device=defaults.device)

max_log_y,y_range
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[400,200], ps=[0.001,0.01],emb_drop=0.04, y_range=y_range, metrics=exp_rmspe)
#learn.model
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(25, 1e-1,wd=0.2)
learn.recorder.plot_losses()
learn.lr_find()

learn.recorder.plot(suggestion = True)
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-3),wd=0.2)
learn.recorder.plot_losses()
#learn.lr_find()

#learn.recorder.plot(suggestion = True)
#learn.unfreeze()

#learn.fit_one_cycle(5, slice(1e-3),wd=0.2)
#learn.recorder.plot_losses()
# get predictions

preds, targets = learn.get_preds(DatasetType.Test)
labels = [np.exp(p[0].data.item()) for p in preds]
submission = pd.DataFrame({'Id': test_id, 'SalePrice': labels})

submission.to_csv('submission.csv', index=False)

submission.head()