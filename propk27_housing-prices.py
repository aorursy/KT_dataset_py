# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#import warnings

#warnings.filterwarnings("ignore", category=DeprecationWarning)

# Any results you write to the current directory are saved as output.
input_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

#input_df.info()
# 1 Try One

#most basic data cleanup - mapping categories to integers,  filling nulls appropriately

data=input_df.copy()



# very low Correlation with saleprice (use df.corr() to get correlation matrix)

drop_list = ['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', '3SsnPorch', 'MiscVal', 'MoSold',

 'YrSold']



# low correlation

drop_list = drop_list + ['ScreenPorch', 'PoolArea']



# no droping for now

drop_list = []



index=data['Id']

data.index=index

colLabels=data.columns.values



train_data = data.copy()

"""

for x in dict_var_list :

    train_data[x]=train_data[x].fillna(0).replace(eval(x))

"""

elec_mode = train_data['Electrical'].mode().values[0]

data['Electrical'].fillna(elec_mode, inplace = True)

global_dict={}

obj_list=list(item for item in colLabels if train_data[item].dtype==np.object)

for x in obj_list:

    global_dict[x]=dict((a,b) for (b,a) in enumerate(set(train_data[x])))

    train_data[x].replace(global_dict[x],inplace=True)

    

train_data['GarageYrBlt'].fillna(0,inplace = True)

train_data['GarageYrBlt'] = train_data['GarageYrBlt'].astype('int64')

train_data['LotFrontage'].fillna(0, inplace = True)

train_data['LotFrontage'] = train_data['LotFrontage'].astype('int64')

train_data['MasVnrArea'].fillna(0, inplace = True)

train_data['MasVnrArea'] = train_data['MasVnrArea'].astype('int64')



train_data = train_data.reset_index(drop = True)

train_data.drop('Id', axis = 1, inplace = True)

target = train_data['SalePrice'].values.ravel()

train = train_data.drop(['SalePrice'], axis = 1)

train.drop(drop_list, axis = 1, inplace = True)

#train.info()
data = test_df.copy()

index=data['Id']

data.index=index

colLabels=data.columns.values



train_data = data.copy()

"""

for x in dict_var_list :

    train_data[x]=train_data[x].fillna(0).replace(eval(x))

"""

elec_mode = train_data['Electrical'].mode().values[0]

data['Electrical'].fillna(elec_mode, inplace = True)

global_dict={}

obj_list=list(item for item in colLabels if train_data[item].dtype==np.object)

for x in obj_list:

    global_dict[x]=dict((a,b) for (b,a) in enumerate(set(train_data[x])))

    train_data[x].replace(global_dict[x],inplace=True)



#no garage, equivalent to garage built long time ago (0 AD) - reconsider value

train_data['GarageYrBlt'].fillna(0,inplace = True)

train_data['LotFrontage'].fillna(0, inplace = True)

train_data['MasVnrArea'].fillna(0, inplace = True)



neg_inf = - (1 << 30)

#Basement doesnot exists so -inf (0 is for unfinished basement)

train_data['BsmtFinSF1'].fillna( neg_inf, inplace = True)

train_data['BsmtFinSF2'].fillna( neg_inf, inplace = True)

train_data['BsmtUnfSF'].fillna( neg_inf, inplace = True)

train_data['TotalBsmtSF'].fillna( neg_inf, inplace = True)



train_data['BsmtFullBath'].fillna(0, inplace = True)

train_data['BsmtHalfBath'].fillna(0, inplace = True)



#no garage, no car space (check if correct)

train_data['GarageCars'].fillna(0, inplace = True)

train_data['GarageArea'].fillna(0, inplace = True)



train_data = train_data.astype('int64')

train_data = train_data.reset_index(drop = True)

train_data.drop('Id', axis = 1, inplace = True)

test = train_data.copy()

test.drop(drop_list, axis = 1, inplace = True)

#test.info()
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

rf.fit(train, target)

pred = rf.predict(test)

print(rf.score(train,target))



np.savetxt('srand_forest.csv', np.c_[range(1461,len(test)+1461),pred], delimiter=',', header = 'Id,SalePrice', comments = '', fmt='%d')



# Edit1: best achieved from this method (no feature engineering) -> 0.17 score



# Edit2: Dropping less related colums (low corr 9 colums) -> 0.20 score - increases performance

# but bad score - better to drop only if it has many null values and outliers
target = np.log10(target)



from sklearn import ensemble

gboost = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

gboost.fit(train, target)

pred2 = gboost.predict(test)

print(gboost.score(train,target))

print(np.power(10, pred2))

np.savetxt('GBoost.csv', np.c_[range(1461,len(test)+1461),np.power(10, pred2)], delimiter=',', header = 'Id,SalePrice', comments = '', fmt='%d,%f')

# upto 0.14 score -> nice! (still long way to go)