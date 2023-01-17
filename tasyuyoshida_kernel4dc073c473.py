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
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import PowerTransformer
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_x = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train.columns
train.describe
train["SalePrice"].describe()
plt.figure(figsize=(20, 10))

sns.distplot(train['SalePrice'])
print("歪度: %f" % train['SalePrice'].skew())

print("尖度: %f" % train['SalePrice'].kurt())
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]



plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)



plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)



data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)



plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
var = '1stFlrSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'MoSold'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(25, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=900000);

plt.xticks(rotation=90);

sns.countplot(x = 'FullBath', hue = "SalePrice", data = train)

plt.legend(loc = "upper right", title = "SalePrice ~ FullBath")
train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



all_data = pd.concat([train_x,test_x],axis=0,sort=True)



train_ID = train['Id']

test_ID = test_x['Id']



all_data.drop("Id", axis = 1, inplace = True)



print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na
plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)
na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()



all_data[na_col_list].dtypes.sort_values()
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#float

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#object

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float to 0

all_data[float_list] = all_data[float_list].fillna(0)



#object to "None"

all_data[obj_list] = all_data[obj_list].fillna("None")



#configure

all_data.isnull().sum()[all_data.isnull().sum() > 0]
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
train_y = np.log1p(train_y)



plt.figure(figsize=(20, 10))

sns.distplot(train_y)
num_feats = all_data.dtypes[all_data.dtypes != "object" ].index



skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)



plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)
skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index



for i in skewed_feats_over:

    print(min(all_data[i]))
pt = PowerTransformer()

pt.fit(all_data[skewed_feats_over])



all_data[skewed_feats_over] = pt.transform(all_data[skewed_feats_over])



#歪度

skewed_feats_fixed = all_data[skewed_feats_over].apply(lambda x: x.skew()).sort_values(ascending = False)



plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats_fixed.index, y=skewed_feats_fixed)
all_data["FeetPerRoom"] =  all_data["TotalSF"]/all_data["TotRmsAbvGrd"]



all_data['YearBuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']



all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])



all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data.dtypes.value_counts()
cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()



train_x[cal_list].info()
all_data = pd.get_dummies(all_data,columns=cal_list)



all_data.shape
train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)

test_x = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)



print("train_x: "+str(train_x.shape))

print("test_x: "+str(test_x.shape))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
train_x, valid_x, train_y, valid_y = train_test_split(

        train_x,

        train_y,

        test_size=0.3,

        random_state=0)
dtrain = xgb.DMatrix(train_x, label=train_y)

dvalid = xgb.DMatrix(valid_x,label=valid_y)



#GBDT

num_round = 5000

evallist = [(dvalid, 'eval'), (dtrain, 'train')]



evals_result = {}



param = {

            'max_depth': 3,

            'eta': 0.01,

            'objective': 'reg:squarederror',

}



bst = xgb.train(

                        param, dtrain,

                        num_round,

                        evallist,

                        evals_result=evals_result,

                        early_stopping_rounds=1000)
plt.figure(figsize=(20, 10))

train_metric = evals_result['train']['rmse']

plt.plot(train_metric, label='train rmse')

eval_metric = evals_result['eval']['rmse']

plt.plot(eval_metric, label='eval rmse')

plt.grid()

plt.legend()

plt.xlabel('rounds')

plt.ylabel('rmse')

plt.ylim(0, 0.3)

plt.show()
ax = xgb.plot_importance(bst)

fig = ax.figure

fig.set_size_inches(10, 30)
dtest = xgb.DMatrix(test_x)

my_submission = pd.DataFrame()

my_submission["Id"] = test_ID

my_submission["SalePrice"] = np.exp(bst.predict(dtest))

# submission

my_submission.to_csv('submission.csv', index=False)