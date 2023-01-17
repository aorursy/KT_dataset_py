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
train = pd.read_csv('/kaggle/input/bangalore-house-prices/Train.csv')

test = pd.read_csv('/kaggle/input/housepricetestdata/Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
train.dtypes
for col in train.columns:

    print(col+" :- ")

    print(train[col].unique())
#Area_Type Train

area_type_dummies = pd.get_dummies(train.area_type)

train = pd.concat([train,area_type_dummies],axis=1)

train = train.drop('area_type',axis=1)
train.head()
#Area_Type Test

area_type_dummies = pd.get_dummies(test.area_type)

test = pd.concat([test,area_type_dummies],axis=1)

test = test.drop('area_type',axis=1)

test.head()
#Availability

def ready_to_move(shift):

    if shift == 'Ready To Move':

        return 1

    else:

        return 0

def immediate_possession(shift):

    if shift == 'Immediate Possession':

        return 1

    else:

        return 0
train['Ready To Move'] = train['availability'].apply(ready_to_move)

train['Immediate Possession'] = train['availability'].apply(immediate_possession)
train.head()
def mon(v):

    try:

        return v.split('-')[1]

    except IndexError:

        pass
train['day'] = train['availability'].apply(lambda v: v.split('-')[0] if len(v.split('-')[0])<4 else 0)



train['Month'] = train['availability'].apply(mon)



train = train.drop('availability',axis=1)
train.head()
test['Ready To Move'] = test['availability'].apply(ready_to_move)

test['Immediate Possession'] = test['availability'].apply(immediate_possession)
test['day'] = test['availability'].apply(lambda v: v.split('-')[0] if len(v.split('-')[0])<4 else 0)



test['Month'] = test['availability'].apply(mon)



test = test.drop('availability',axis=1)
test.head()
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

train['location_encode']= label_encoder.fit_transform(train['location'].astype(str)) 

train.head()
test['location_encode']= label_encoder.fit_transform(test['location'].astype(str)) 
test.head()
train = train.drop('location',axis=1)

test = test.drop('location',axis=1)
train['size'].isnull().sum()
train['size'].mode()
train['size'].fillna("2 BHK",inplace=True)
train['size'].isnull().sum()
def bhk(v):

    try:

        if v.split(' ')[1]=='BHK':

            return v.split(' ')[0]

        else:

            return 0

    except IndexError:

        pass



def bedroom(v):

    try:

        if v.split(' ')[1]=='Bedroom':

            return v.split(' ')[0]

        else:

            return 0

    except IndexError:

        pass
train['BHK'] = train['size'].apply(bhk)

train['Bedroom'] = train['size'].apply(bedroom)
train.head()
test['size'].isnull().sum()
test['size'].mode()
test['size'].fillna("2 BHK",inplace=True)
test['size'].isnull().sum()
test['BHK'] = test['size'].apply(bhk)

test['Bedroom'] = test['size'].apply(bedroom)
test.head()
train = train.drop('size',axis=1)

test = test.drop('size',axis=1)
train['society'].isnull().sum()
train['society'].fillna("No Society",inplace=True)
label_encoder_soc = preprocessing.LabelEncoder() 

  

train['society_encode']= label_encoder_soc.fit_transform(train['society'].astype(str))
train.head()
test['society'].isnull().sum()
test['society'].fillna("No Society",inplace=True)
test['society_encode']= label_encoder_soc.fit_transform(test['society'].astype(str))
test.head()
train = train.drop('society',axis=1)

test = test.drop('society',axis=1)
train['total_sqft'].unique()
train['total_sqft'].isnull().sum()
train['total_sqft'] = train['total_sqft'].str.extract('(\d+)', expand=False)
train[train.total_sqft.str.len()<3]
train['total_sqft_1'] = train['total_sqft'].apply(lambda v: str(int(v)*9) if len(v)==2 else v)
train['total_sqft_1'] = train['total_sqft_1'].apply(lambda v: str(int(v)*43560) if len(v)<2 else v)
train[train.total_sqft_1.str.len()<3]
train['total_sqft_1'] = train['total_sqft_1'].astype(int)
train['total_sqft'] = train['total_sqft_1']

train = train.drop('total_sqft_1',axis=1)
train.head()
test['total_sqft'].isnull().sum()
test['total_sqft'] = test['total_sqft'].str.extract('(\d+)', expand=False)
test['total_sqft_1'] = test['total_sqft'].apply(lambda v: str(int(v)*9) if len(v)==2 else v)



test['total_sqft_1'] = test['total_sqft_1'].apply(lambda v: str(int(v)*43560) if len(v)<2 else v)
test['total_sqft_1'] = test['total_sqft_1'].astype(int)



test['total_sqft'] = test['total_sqft_1']



test = test.drop('total_sqft_1',axis=1)
test.head()
train['bath'].isnull().sum()
train['bath'].mode()
train['bath'].fillna(2.0,inplace=True)
test['bath'].isnull().sum()
test['bath'].mode()
test['bath'].fillna(2.0,inplace=True)
train['balcony'].isnull().sum()
train['balcony'].mode()
train['balcony'].fillna(2.0,inplace=True)
test['balcony'].isnull().sum()
test['balcony'].mode()
test['balcony'].fillna(1.0,inplace=True)
train.head()
test.head()
train.isnull().sum()
train['Month'].fillna("No Month", inplace=True)
train['Month'].isnull().sum()
test.isnull().sum()
test['Month'].fillna("No Month", inplace=True)
test['Month'].isnull().sum()
label_encoder_mon = preprocessing.LabelEncoder() 

  

train['month_encode']= label_encoder_mon.fit_transform(train['Month'].astype(str))

test['month_encode']= label_encoder_mon.fit_transform(test['Month'].astype(str))
train.head()
test.head()
train = train.drop('Month',axis=1)

test = test.drop('Month',axis=1)
train['BHK'] = train['BHK'].astype(int)

test['BHK'] = test['BHK'].astype(int)

train['Bedroom'] = train['Bedroom'].astype(int)

test['Bedroom'] = test['Bedroom'].astype(int)

train['day'] = train['day'].astype(int)

test['day'] = test['day'].astype(int)
X = train.drop('price',axis=1)

y = train['price']

test_X = test.drop('price',axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
import xgboost as xgb

xg_reg = xgb.XGBRegressor(colsample_bytree = 0.6, min_child_weight=10, learning_rate = 0.07, max_depth = 12, n_estimators = 500)
xg_reg.fit(X_train,y_train)



preds = abs(xg_reg.predict(X_test))
1 - np.sqrt(np.square(np.log10(preds +1) - np.log10(y_test +1)).mean())
import lightgbm as lgb

model_lgb=lgb.LGBMRegressor(bagging_fraction=0.8, bagging_frequency=4, boosting_type='gbdt', colsample_bytree=1.0, feature_fraction=0.5, learning_rate=0.1, max_depth=30,

              min_child_samples=20, min_child_weight=30, n_estimators=500,

              num_leaves=1200)



model_lgb.fit(X_train,y_train)

pred_lgb = abs(xg_reg.predict(X_test))
1 - np.sqrt(np.square(np.log10(pred_lgb +1) - np.log10(y_test +1)).mean())
my_submission = pd.DataFrame({'price': preds})

my_submission.to_csv('submission_1.csv', index=False)
# Got a point of .80 in leader Board.