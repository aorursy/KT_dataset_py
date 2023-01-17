import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
for dirname, _, filenames in os.walk('/kaggle/input/home-data-for-ml-course'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
### Read train and test data
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")
train.head()
train.dtypes
pd.set_option('display.max_rows', 500)
missing = train.isna().sum()*100/train.shape[0]
missing[missing!=0]
def preprocess_data(df):
    ## Drop columns with high missing rate
    df = df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1)
    ## Drop not so useful columns 
    df = df.drop(['GarageYrBlt','Utilities', 'Street'],axis = 1)
    ## Change datatype
    df[['MSSubClass','YrSold', 'MoSold']] = df[['MSSubClass','YrSold', 'MoSold']].astype(str)
    ## Fill missing values for categorical columns 
    for col in ['GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType',
                'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
               'Electrical']:
        df[col] = df[col].fillna(df[col].mode())
    ## Fill missing values for numerical columns
    for col in ['MasVnrArea']:
        df[col] = df[col].fillna(df[col].mean())
    
    df = pd.get_dummies(df).reset_index(drop=True)
    return df
y = np.log1p(train["SalePrice"])
train = train.drop('SalePrice',axis = 1)
full_data = preprocess_data(pd.concat([train, test]).reset_index(drop=True))
X = full_data.iloc[:len(y),:]
test = full_data.iloc[len(y):,:]
print(f"x Shape: {X.shape}")
print(f"Test Shape: {test.shape}")
print(f"Y Shape: {y.shape}")
# rmsle
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=1000,
                       max_depth=3, min_child_weight=1,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.01,
                       eval_set=[(X_train, y_train), (X_val, y_val)],
                       verbose = True)
xgboost.fit(X_train,y_train)
val_preds = xgboost.predict(X_val)
rmsle(y_val, val_preds)
submission = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")

submission.iloc[:, 1] = np.floor(np.expm1(xgboost.predict(test)))

submission.to_csv("House_price_submission.csv", index=False)
