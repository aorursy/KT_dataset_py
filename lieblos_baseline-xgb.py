import numpy as np
import pandas as pd
import os

INPUT_DIR = '../input'
print(os.listdir(INPUT_DIR))
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
X_train, y_train = train.drop('SalePrice', axis=1), train['SalePrice']
X_test = test
X_train.drop('Id', axis=1, inplace=True)
X_test.drop('Id', axis=1, inplace=True)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
train.head()
test.head()
import seaborn as sns

sns.scatterplot(x=X_train['1stFlrSF'], y=y_train)
from scipy import stats
from scipy.stats import norm, skew

sns.distplot(y_train, fit=norm)
X_train_samples = X_train.shape[0]
X_test_samples = X_test.shape[0]
merged = pd.concat([X_train, X_test])
merged.head()
import missingno as mn
columns_with_missingnos = [col for col in list(merged) if merged[col].isnull().values.any()]
mn.matrix(merged[columns_with_missingnos], labels=True)
merged[columns_with_missingnos].info()
categorical_columms_with_missingno = [col for col in columns_with_missingnos if merged[col].dtype == 'O']
numerical_columms_with_missingno = [col for col in columns_with_missingnos if merged[col].dtype == 'float64']
for col in numerical_columms_with_missingno:
    merged[col] = merged[col].fillna(0.)
for col in categorical_columms_with_missingno:
    merged[col] = merged[col].fillna('None')
import matplotlib.pyplot as plt

plt.subplots(figsize=(12,12))
sns.heatmap(merged.corr(), vmax=0.9, square=True)
from sklearn.preprocessing import LabelEncoder

categorical_columns = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
    'GarageCond', 'PavedDrive','PoolQC', 'Fence', 'MiscFeature', 'SaleType', 
    'SaleCondition']

for col in categorical_columns:
    merged[col] = merged[col].astype('category')
    
merged[categorical_columns] = merged[categorical_columns].apply(LabelEncoder().fit_transform)
merged.head()
X_train, X_test = merged[:X_train_samples], merged[X_train_samples:]
import xgboost as xgb

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X_train, y_train)
preds = model_xgb.predict(X_test)
submission = pd.DataFrame({
    'Id': range(1461, X_test_samples + 1461),
    'SalePrice': preds,
})
display(sample_submission.head(2))
display(submission.head(2))
submission.to_csv('submission.csv', index=False)
