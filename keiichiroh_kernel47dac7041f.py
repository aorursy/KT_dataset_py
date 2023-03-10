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
#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.describe
train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() # ???????????????????????????????????????

alldata[na_col_list].dtypes.sort_values() #????????????
na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist() #float64

na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist() #object

# float64?????????????????????????????????0?????????

for na_float_col in na_float_cols:

    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

# object?????????????????????????????????'NA'?????????

for na_obj_col in na_obj_cols:

    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'
alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
# ???????????????????????????????????????????????????

cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()

# ???????????????????????????????????????

num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()

# ?????????????????????????????????????????????????????????????????????

other_cols = ['Id','WhatIsData']

# ???????????????????????????????????????

cat_cols.remove('WhatIsData') #?????????????????????????????????????????????????????????

num_cols.remove('Id') #Id??????

# ???????????????????????????????????????

alldata_cat = pd.get_dummies(alldata[cat_cols])

# ???????????????

all_data = pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)
choice_list=['GrLivArea','GarageCars','OverallQual']

data=train.loc[:,choice_list]
data
train_df = pd.DataFrame(data)

train_df['SalePrice'] = train.SalePrice
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

X = train_df[['GrLivArea','GarageCars','OverallQual']].values         # ???????????????Numpy????????????

Y = train_df['SalePrice'].values         # ???????????????Numpy????????????



lr.fit(X, Y)                         # ?????????????????????????????????
print('coefficient = ', lr.coef_[0]) # ??????????????????????????????

print('intercept = ', lr.intercept_) # ???????????????

'''

plt.scatter(X, Y, color = 'blue')         # ?????????????????????????????????????????????????????????????????????

plt.plot(X, lr.predict(X), color = 'red') # ???????????????????????????



plt.title('??????????????????')               # ??????????????????

plt.xlabel('Average number of rooms [RM]') # x???????????????

plt.ylabel('Prices in $1000\'s [MEDV]')    # y???????????????

plt.grid()                                 # ????????????????????????



plt.show()                                 # ????????????

'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0) # ??????????????????????????????????????????



lr = LinearRegression()

lr.fit(X_train, Y_train) # ?????????????????????????????????
Y_pred = lr.predict(X_test) # ????????????????????????????????????????????????



plt.scatter(Y_pred, Y_pred - Y_test, color = 'blue')      # ????????????????????? 

plt.hlines(y = 0, xmin = -10, xmax = 50, color = 'black') # x????????????????????????????????????

plt.title('Residual Plot')                                # ??????????????????

plt.xlabel('Predicted Values')                            # x???????????????

plt.ylabel('Residuals')                                   # y???????????????

plt.grid()                                                # ????????????????????????



plt.show()                                               # ????????????
from sklearn.metrics import mean_squared_error



Y_train_pred = lr.predict(X_train) # ????????????????????????????????????????????????

print('MSE train data: ', mean_squared_error(Y_train, Y_train_pred)) # ???????????????????????????????????????????????????????????????

print('MSE test data: ', mean_squared_error(Y_test, Y_pred))         # ???????????????????????????????????????????????????????????????
from sklearn.metrics import r2_score



print('r^2 train data: ', r2_score(Y_train, Y_train_pred))

print('r^2 test data: ', r2_score(Y_test, Y_pred))
# ????????????????????????





sub_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub_tmp = sub_data.loc[:, "Id"]

sub = sub_data.loc[:, choice_list]

sub.isnull().sum()[sub.isnull().sum()>0].sort_values(ascending=False)



sub.dtypes.sort_values() #????????????

na_float_cols = sub.dtypes[sub.dtypes=='float64'].index.tolist() #float64

na_obj_cols = sub.dtypes[sub.dtypes=='object'].index.tolist() #object



print(na_float_cols)

# float64?????????????????????????????????0?????????

for na_float_col in na_float_cols:

    sub.loc[sub[na_float_col].isnull(),na_float_col] = 0.0

# object?????????????????????????????????'NA'?????????

for na_obj_col in na_obj_cols:

    sub.loc[sub[na_obj_col].isnull(),na_obj_col] = 'NA'



test_pred = lr.predict(sub) # ????????????????????????????????????????????????



print('r^2 train data: ', r2_score(Y_train, Y_train_pred))

print('r^2 test data: ', r2_score(Y_test, Y_pred))





my_submission = pd.DataFrame()

my_submission["Id"] = list(map(int, sub_tmp))

my_submission["SalePrice"] = list(map(int, test_pred))

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)