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

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)
train.head(2)
test.head(2)
train.shape , test.shape
#train.set_index('Id',inplace=True)
#test.set_index('Id',inplace=True)
train.rename(str.lower, axis='columns',inplace=True)
test.rename(str.lower, axis='columns',inplace=True)
#train.isnull().sum()
train['alley'].fillna('NO',inplace=True)

train.loc[train.index == 948, 'bsmtexposure'] = 'No'

train.loc[train.index == 332 , 'bsmtfintype2'] = 'Unf'

train.masvnrtype.fillna( value='None' , inplace=True)

train.masvnrarea.fillna( value= 0 , inplace= True)



train['lotfrontage'].fillna(train.lotfrontage.mean(),inplace=True)
train['garageyrblt'].fillna(0,inplace=True)

train['garageyrblt'].astype(int)
train['bsmtqual'].fillna('No Basement' , inplace=True)

train['bsmtfintype2'].fillna('No Basement' , inplace=True)

train['bsmtfintype1'].fillna('No Basement' , inplace=True)

train['bsmtexposure'].fillna('No Basement' , inplace=True)

train['bsmtcond'].fillna('No Basement' , inplace=True)
train['garagefinish'].fillna('No', inplace = True)

train['garagequal'].fillna("No", inplace = True)

train['garagecond'].fillna("No", inplace = True)

train.drop(['poolqc','miscfeature','fence'], inplace=True, axis=1)
train.dropna(subset=['electrical'],inplace=True) # i dropped this to make train and test equal in rows
train.drop(columns=['bsmtfinsf1' , 'bsmtfinsf2' ,'bsmtunfsf'] , axis=1  , inplace=True)
test['alley'].fillna('No',inplace=True)

test['bsmtqual'].fillna('NO',inplace=True)

test['bsmtcond'].fillna('NO',inplace=True)

test['bsmtexposure'].fillna('NO',inplace=True)

test['bsmtfintype1'].fillna('NO',inplace=True)

test['bsmtfintype2'].fillna('NO',inplace=True)

test['fireplacequ'].fillna('NO',inplace=True)

test['garagetype'].fillna('NO',inplace=True)

test['garagefinish'].fillna('NO',inplace=True)

test['garagequal'].fillna('NO',inplace=True)

test['garagecond'].fillna('NO',inplace=True)

test['poolqc'].fillna('NO',inplace=True)

test['fence'].fillna('NO',inplace=True)

test['miscfeature'].fillna('None',inplace=True)
test['garagecars'].fillna(test.garagecars.mean(), inplace=True)

test['garagearea'].fillna(test.garagearea.mean(), inplace=True)

test['totalbsmtsf'].fillna(test.totalbsmtsf.mean(),inplace=True)
test.drop(['poolqc','miscfeature','fence'], inplace=True, axis=1)
test.drop(columns=['bsmtfinsf1' , 'bsmtfinsf2' ,'bsmtunfsf'] , axis=1  , inplace=True)


test['lotfrontage'].fillna(test.lotfrontage.mean(),inplace=True)

test['masvnrarea'].fillna(test.masvnrarea.mean(),inplace=True)

test['bsmtfullbath'].fillna(test.bsmtfullbath.mean(),inplace=True)

test['bsmthalfbath'].fillna(test.bsmthalfbath.mean(),inplace=True)

test['garageyrblt'].fillna(test.garageyrblt.mean(),inplace=True)
train.shape , test.shape
train_dum = pd.get_dummies(train ,drop_first=True)
train_dum.shape
test_dum = pd.get_dummies(test ,drop_first=True)
test_dum.shape
train_dum.equals(test_dum)
df_new = []

for col in train_dum.columns:

    for col2 in test_dum.columns:

        if col == col2 :

            df_new.append(col)

            

df_new

        
train.shape , test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge, Lasso, LinearRegression

from sklearn.model_selection import train_test_split
#do a model with only the numerical vars 

feature_col = ['overallqual','totalbsmtsf','1stflrsf','grlivarea',

               'garagecars' ,'garagearea']
XX = train[feature_col]

y= train.saleprice
ss = StandardScaler()

X_std = ss.fit_transform(XX)
my_model = RandomForestRegressor()

my_model.fit(XX,y)

my_model.score(XX,y)
X_train , x_test , y_train , y_test =train_test_split(XX,y,test_size =.3 , shuffle=True) # by StandardScaler
my_model.score(X_train,y_train)
my_model.score(x_test,y_test)
test_X = test[feature_col]
predicted_prices = my_model.predict(test_X)
predicted_prices
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_RF.csv', index=False)
X_lasso = train_dum[df_new]

y_lasso = train_dum.saleprice
sss = StandardScaler()

X_train , X_test , y_train , y_test = train_test_split(X_lasso , y_lasso  , test_size = .3 , shuffle = True)

sss.fit(X_train)

X_train_std = sss.transform(X_train)

X_test_std = sss.transform(X_test)
ls = Lasso()

ls.fit(X_train_std, y_train)

print(ls.score(X_train_std, y_train))

ls.score(X_test_std, y_test)
test_X_lasso = test_dum[df_new]
predicted_prices_lasso = ls.predict(test_X_lasso)
predicted_prices_lasso
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission_lasso.csv', index=False)
train.reset_index()

test.reset_index(inplace=True)
my_submission = pd.DataFrame({'Id': test.id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission##2.csv', index=False)