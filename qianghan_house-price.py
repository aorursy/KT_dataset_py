# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import copy

from scipy import *

import warnings

warnings.filterwarnings('ignore')



train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

print(train.shape)

train.head()
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(test.shape)

test.head()
#f=open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt", "r")

#desc=f.read()

#desc=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt")
train.describe(include='all')
test.describe(include='all')
print(train.columns,'\n',train.dtypes)
train_ID=train['Id'];test_ID=test['Id']

train_Y=train['SalePrice'];

train_X=train.drop(['Id','SalePrice'],axis=1);test_X=test.drop(['Id'],axis=1)

ntrain=len(train_X)

#print(ntrain)

combine_X=pd.concat([train_X,test_X])

print(combine_X.shape)

combine_X.head()
f, ax = plt.subplots(figsize=(12, 9))

data=combine_X

missing_data=(pd.isnull(data).sum()[pd.isnull(data).sum()!=0]/len(data)).sort_values(ascending=False)

sns.barplot(x=missing_data.index,y=missing_data)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()

missing_data.head(10)
corrmat=train.corr().abs()#['SalePrice']

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat)

corrmat.head()
missing_data.index
temp=combine_X['PoolQC']

print('before imputing',temp.unique())

poolqc_maping={'Ex':4,'Gd':3,'TA':2,'Fa':1}

temp=temp.map(poolqc_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['PoolQC']=temp

#pd.isnull(combine_X['PoolQC']).sum()

temp=combine_X['MiscFeature']

print('before imputing',temp.unique())

temp=temp.fillna("None")

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['MiscFeature']=temp



temp=combine_X['Alley']

print('before imputing',temp.unique())

temp=temp.fillna("None")

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Alley']=temp

temp=combine_X['Fence']

print('before imputing',temp.unique())

Fence_maping={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1}

temp=temp.map(Fence_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Fence']=temp

temp=combine_X['FireplaceQu']

print('before imputing',temp.unique())

FireplaceQu_maping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}

temp=temp.map(FireplaceQu_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['FireplaceQu']=temp

temp=combine_X['LotFrontage']

#print(temp,type(temp))

print('before imputing',temp.unique())

temp[:ntrain]=temp[:ntrain].fillna(int(train['LotFrontage'].mean()))

temp[ntrain:]=temp[ntrain:].fillna(int(test['LotFrontage'].mean()))

print('after imputing',pd.isnull(temp).sum())

combine_X['LotFrontage']=temp

temp=combine_X['GarageFinish']

print('before imputing',temp.unique())

GarageFinish_maping={'Fin':3,'RFn':2,'Unf':1}

temp=temp.map(GarageFinish_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageFinish']=temp

temp=combine_X['GarageYrBlt']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp[:ntrain]=temp[:ntrain].fillna(train['GarageYrBlt'].mode()[0])

temp[ntrain:]=temp[ntrain:].fillna(test['GarageYrBlt'].mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageYrBlt']=temp

temp=combine_X['GarageQual']

print('before imputing',temp.unique())

GarageQual_maping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}

temp=temp.map(GarageQual_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageQual']=temp

temp=combine_X['GarageCond']

print('before imputing',temp.unique())

GarageCond_maping={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}

temp=temp.map(GarageCond_maping)

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageCond']=temp

temp=combine_X['GarageType']

print('before imputing',temp.unique())

temp=temp.fillna("None")

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageType']=temp

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    temp=combine_X[col]

    print(col,'before imputing',temp.unique())

    temp=temp.fillna(0)

    print(col,'after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

    combine_X[col]=temp



  
mapping=[{'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},

         {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1},

         {'Gd':4,'Av':3,'Mn':2,'No':1},

         {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1},

         {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1}

         ]

for idx,col in enumerate(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']):

    temp=combine_X[col]

    print(col,'before imputing',temp.unique())

    temp=temp.map(mapping[idx])

    temp=temp.fillna(0)

    print(col,'after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

    combine_X[col]=temp

temp=combine_X['MasVnrType']

print('before imputing',temp.unique())

temp=temp.fillna("None")

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['MasVnrType']=temp

temp=combine_X['MasVnrArea']

print('before imputing',temp.unique())

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['MasVnrArea']=temp

temp=combine_X['MSZoning']

print('before imputing',temp.unique(),'\n',temp.value_counts())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['MSZoning']=temp

temp=combine_X['Utilities']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

combine_X=combine_X.drop(['Utilities'],axis=1)



#temp.head()

#type(temp)

#temp.plot()

#sns.distplot(temp)
temp=combine_X['Functional']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n','sum=',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Functional']=temp



temp=combine_X['Exterior2nd']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Exterior2nd']=temp
temp=combine_X['Exterior1st']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Exterior1st']=temp
temp=combine_X['SaleType']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['SaleType']=temp
temp=combine_X['Electrical']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['Electrical']=temp
temp=combine_X['KitchenQual']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(temp.mode()[0])

KitchenQual_maping={'Ex':4,'Gd':3,'TA':2,'Fa':1}

temp=temp.map(KitchenQual_maping)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['KitchenQual']=temp
temp=combine_X['GarageCars']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageCars']=temp
temp=combine_X['GarageArea']

print('before imputing',temp.unique(),'\n',temp.value_counts()/len(temp),

      '\n',(temp.value_counts()/len(temp)).sum())

temp=temp.fillna(0)

print('after imputing',temp.unique(),'\n',pd.isnull(temp).sum())

combine_X['GarageArea']=temp
data=combine_X

missing_data=(pd.isnull(data).sum()[pd.isnull(data).sum()!=0]/len(data)).sort_values(ascending=False)

missing_data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#count # of unique value in categorical features 

combine_X['ExterQual']=combine_X['ExterQual'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1})

combine_X['ExterCond']=combine_X['ExterCond'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1})

combine_X['HeatingQC']=combine_X['HeatingQC'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1})

Ncat_cut=30

for cat in combine_X.dtypes[combine_X.dtypes=='object'].index:

    if (combine_X[cat]).nunique()==2:

         le=LabelEncoder()

         combine_X[cat]=le.fit_transform(combine_X[cat])

    if  (combine_X[cat]).nunique()>2 and (combine_X[cat]).nunique()<=Ncat_cut:

        le=LabelEncoder()

        feature=le.fit_transform(combine_X[cat])

        feature = array(feature).reshape(combine_X.shape[0], 1)

        ohe = OneHotEncoder(sparse=False, categories='auto')

        feature = ohe.fit_transform(feature)

        cols=[]

        for i in range(feature.shape[1]):

            cols.append(cat+'_'+str(i))

        combine_X=combine_X.join(pd.DataFrame(feature,columns=cols))

        combine_X=combine_X.drop([cat],axis=1)

    elif  (combine_X[cat]).nunique()>Ncat_cut:

        combine_X=combine_X.drop([cat],axis=1)

        

print(combine_X.shape)

print(combine_X.head())

#for item in combine_X.columns:

#    print(item)

print('total # of categorical features:',len(combine_X.dtypes[combine_X.dtypes=='object']))



 
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import make_scorer,mean_squared_error

from xgboost import XGBRegressor



train_X=combine_X[:ntrain]

test_X=combine_X[ntrain:]

#train_Y=log(train_Y)



params = {

        'n_estimators': [60, 100, 150],

        'reg_alpha': [0,0.5, 1, 1.5, 2],

        'max_depth': [3, 4, 5, 6]

        }

xgb = XGBRegressor(learning_rate=0.05)

print(xgb)

grid = GridSearchCV(xgb, 

                    param_grid = params, 

                    scoring = make_scorer(mean_squared_error,greater_is_better=False), 

                    n_jobs = -1, 

                    cv = 5,)



reg=grid.fit(train_X,train_Y)

print('score=',reg.score(train_X,train_Y))



predictions = reg.predict(test_X)

print('preds=',predictions)

output=pd.DataFrame({"Id":test_ID,'SalePrice':predictions})

output.to_csv('submission.csv', index=False)

print("The submission was successfully saved!")

##-5032789641.283177
from scipy.stats import norm, skew #for some statistics

sns.distplot(train['SalePrice'], fit=norm);

sns.distplot(log(train['SalePrice']), fit=norm);
