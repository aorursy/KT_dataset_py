import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/home-data-for-ml-course/train.csv")

test_df = pd.read_csv("../input/home-data-for-ml-course/test.csv")
full_df = pd.concat([train_df,test_df],keys=[0,1])
full_df.columns
full_df['Alley'] = full_df['Alley'].fillna("No Alley")

full_df['BsmtQual'] = full_df['BsmtQual'].fillna("No Basement")

full_df['BsmtCond'] = full_df['BsmtCond'].fillna("No Basement")

full_df['BsmtExposure'] = full_df['BsmtExposure'].fillna("No Basement")

full_df['BsmtFinType1'] = full_df['BsmtFinType1'].fillna("No Basement")

full_df['BsmtFinType2'] = full_df['BsmtFinType2'].fillna("No Basement")

full_df['FireplaceQu'] = full_df['FireplaceQu'].fillna("No Fireplace")

full_df['GarageType'] = full_df['GarageType'].fillna("No Garage")

full_df['GarageFinish'] = full_df['GarageFinish'].fillna("No Garage")

full_df['GarageQual'] = full_df['GarageQual'].fillna("No Garage")

full_df['GarageCond'] = full_df['GarageCond'].fillna("No Garage")

full_df['PoolQC'] = full_df['PoolQC'].fillna("No Pool")

full_df['Fence'] = full_df['Fence'].fillna("No Fence")

full_df['MiscFeature'] = full_df['MiscFeature'].fillna("No MiscFeature")
missing = [col for col in full_df.columns if full_df[col].isnull().values.any()]

missing
categ_missing = [col for col in missing if full_df[col].dtype == 'O']

categ_missing
for col in categ_missing:

    full_df[col] = full_df[col].fillna(full_df[col].mode()[0])
#Deal with ordinal features here at a future date...
#use knn imputer on numeric columns

numeric = [col for col in full_df.columns if full_df[col].dtype != 'O']

numeric.remove('Id')

numeric.remove('SalePrice')

numeric_missing = [col for col in missing if full_df[col].dtype != 'O']

numeric_missing.remove('SalePrice')

numeric_missing
#label encode categorical columns...

categs = [col for col in full_df.columns if full_df[col].dtype == 'O']

from sklearn.preprocessing import LabelEncoder

full_df[categs] = full_df[categs].apply(LabelEncoder().fit_transform)
#use knn imputer on numeric columns

#numeric_missing defined before label-encoding of categorical 
num_df = full_df[numeric].copy(deep=True)
num_df
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_df=pd.DataFrame(scaler.fit_transform(num_df),columns=num_df.columns)
num_df
from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=5)

knn.fit(num_df)

num_df = pd.DataFrame(knn.transform(num_df),columns=num_df.columns)
num_df
import seaborn as sns

sns.heatmap(num_df.isnull(), cbar=False)
full_df.drop(labels=numeric, axis="columns", inplace=True)

for col in numeric:

    full_df[col] = num_df[col].values
full_df
####################

train,test = full_df.xs(0),full_df.xs(1)

test.drop('SalePrice', axis=1,inplace=True)
train = train.set_index('Id')

test = test.set_index('Id')
len(X.columns)
X = train.drop('SalePrice',axis=1)

y = train['SalePrice']
from sklearn.feature_selection import SelectKBest, chi2

KBest = SelectKBest(chi2, k=77)

KBest.fit(X, y)

mask = KBest.get_support()

X_new = KBest.transform(X)

new_features = X.columns[mask]

final_train = pd.DataFrame(X_new,columns=new_features)
final_train['SalePrice'] = train['SalePrice'].values
final_test = test[new_features]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice',axis=1),train['SalePrice'], test_size=0.33, random_state=42)
#XGBoost hyper-parameter tuning

#credits to https://www.kaggle.com/felipefiorini/xgboost-hyper-parameter-tuning

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

def hyperParameterTuning(X_train, y_train):

    param_tuning = {

        'learning_rate': [0.01, 0.1],

        'max_depth': [3, 5, 7, 10],

        'min_child_weight': [1, 3, 5],

        'subsample': [0.5, 0.7],

        'colsample_bytree': [0.5, 0.7],

        'n_estimators' : [100, 200, 500],

        'objective': ['reg:squarederror']

    }



    xgb_model = xgb.XGBRegressor()



    gsearch = GridSearchCV(estimator = xgb_model,

                           param_grid = param_tuning,                        

                           #scoring = 'neg_mean_absolute_error', #MAE

                           scoring = 'neg_mean_squared_error',  #MSE

                           cv = 5,

                           n_jobs = -1,

                           verbose = 1)



    gsearch.fit(X_train,y_train)



    return gsearch.best_params_
#stri = hyperParameterTuning(X_train, y_train)

#print(stri)
best_params = {

    'colsample_bytree': 0.5,

    'learning_rate': 0.1, 

    'max_depth': 5,

    'min_child_weight': 1, 

    'n_estimators': 500, 

    'objective': 

    'reg:squarederror', 

    'subsample': 0.7}
#Now ready to use XGBoost to create predictions

import xgboost as xgb

regressor = xgb.XGBRegressor(    colsample_bytree= 0.5,

    learning_rate= 0.1, 

    max_depth= 5,

    min_child_weight= 1, 

    n_estimators= 500, 

    objective= 'reg:squarederror', 

    subsample= 0.7)

regressor.fit(final_train.drop('SalePrice',axis=1),final_train['SalePrice'])

preds =regressor.predict(final_test)

submission = pd.DataFrame(data=(preds) ,columns=['SalePrice'])

submission['Id']=final_test.index

submission.to_csv('submission.csv', index=False)
submission