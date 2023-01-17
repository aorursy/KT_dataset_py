import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path_train = '../input/train.csv'
path_test = '../input/train.csv'
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df = df_train
df.head()
df.info()
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
def lotfrontage_cleaner():
    neighborhoods = df['Neighborhood'].drop_duplicates().reset_index(drop=True)

    for neighborhood in neighborhoods:
        median = df['LotFrontage'].loc[df['Neighborhood'] == neighborhood].median()
        df['LotFrontage'].loc[df['Neighborhood'] == neighborhood] = df['LotFrontage'].loc[df['Neighborhood'] == neighborhood].fillna(median)
#define function to filter df by only the null columns in a chosen column
def null_rowsincolumn(df, null_columns, chosen_column):
    
    filtered_df = df[null_columns][df[null_columns][chosen_column].isnull()]
    return(filtered_df)
df['Alley'].value_counts()
#define function to replace non null values with Alley & null values with no_alley
def alley_cleaner():
    df.loc[df['Alley'].notnull(), 'Alley'] = 'alley'
    df['Alley'].fillna('no_alley', inplace=True)
null_rowsincolumn(df, null_columns, 'MasVnrType')
#define function to replace NA in MasVnrType with none & in MasVnrArea with 0
def MasVnr_cleaner():
    df['MasVnrType'].fillna('None', inplace=True)
    df['MasVnrArea'].fillna(0 , inplace=True)
null_rowsincolumn(df, null_columns, 'BsmtExposure')
#define function to drop any row where not all bsmt attributes are NA 
def bsmt_cleaner():
    columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    df.drop(df[(df[columns].isnull().sum(axis=1) < len(columns)) & (df[columns].isnull().sum(axis=1) > 0)].index, inplace=True)
    df[columns] = df[columns].fillna(value = 'no_basement')
def fireplace_cleaner():
    df['FireplaceQu'][(df['FireplaceQu'].isnull()) & (df['Fireplaces'] == 0)] = df['FireplaceQu'][(df['FireplaceQu'].isnull()) & (df['Fireplaces'] == 0)].fillna('None')
#define function to drop any row where not all bsmt attributes are NA 
def garage_cleaner():
    columns = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
    #fill all garage attributes (in above column array) with none for rows where Garage Area = 0
    df[columns][df['GarageArea'] == 0] = df[columns][df['GarageArea'] == 0].fillna(value = 'none')
    df.drop(df[(df[columns].isnull().sum(axis=1) < len(columns)) & (df[columns].isnull().sum(axis=1) > 0)].index, inplace=True)
    df[columns] = df[columns].fillna(value = 'none')
def pool_cleaner():
    
    df['PoolQC'][df['PoolArea'] == 0] = df['PoolQC'][df['PoolArea'] == 0].fillna(value = 'None')
    df['PoolQC'].fillna(value='None', inplace=True)
def fence_cleaner():
    
    df['Fence'].fillna('none', inplace=True)
def misc_cleaner():
    
    df['MiscFeature'].fillna('none', inplace=True)
def final_cleaner():
    
    df.fillna(value=0, inplace=True)
lotfrontage_cleaner()
alley_cleaner()
MasVnr_cleaner()
bsmt_cleaner()
fireplace_cleaner()
pool_cleaner()
fence_cleaner()
garage_cleaner()
misc_cleaner()

final_cleaner()
df[null_columns].isnull().sum()
df.hist(bins=30, figsize=(20,15));
plt.tight_layout()
# We will get all numerical columns and assign them to df_columns
df_columns = df._get_numeric_data()
columns = df_columns.columns
columns
for col in columns:
    
    sns.jointplot(x=col, y='SalePrice', kind='reg', data=df);
    plt.show()

#drop irrelevant columns:
columns_drop = ['PoolArea', 'MiscVal', 'MoSold', 'YrSold', '3SsnPorch', 'EnclosedPorch', 'BsmtHalfBath', 
                'LowQualFinSF', 'BsmtFinSF2']
df.drop(columns = columns_drop, inplace=True)
df['2ndFlrSF'][df['2ndFlrSF'] > 0] = 1
df['2ndFlrSF'][df['2ndFlrSF'] == 0] = 0
df['OpenPorchSF'][df['OpenPorchSF'] > 0] = 1
df['OpenPorchSF'][df['OpenPorchSF'] == 0] = 0
df['ScreenPorch'][df['ScreenPorch'] > 0] = 1
df['ScreenPorch'][df['ScreenPorch'] == 0] = 0
df['WoodDeckSF'][df['WoodDeckSF'] > 0] = 1
df['WoodDeckSF'][df['WoodDeckSF'] == 0] = 0
df.set_index('Id', inplace=True)
df.head()
#Create a dictionary with the values & their interpretations:
subclass_dict = {
    20: '1-STORY 1946 & NEWER ALL STYLES',
    30: '1-STORY 1945 & OLDER',
    40: '1-STORY W/FINISHED ATTIC ALL AGES',
    45: '1-1/2 STORY - UNFINISHED ALL AGES',
    50: '1-1/2 STORY FINISHED ALL AGES',
    60: '2-STORY 1946 & NEWER',
    70: '2-STORY 1945 & OLDER',
    75: '2-1/2 STORY ALL AGES',
    80: 'SPLIT OR MULTI-LEVEL',
    85: 'SPLIT FOYER',
    90: 'DUPLEX - ALL STYLES AND AGES',
    120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
    150: '1-1/2 STORY PUD - ALL AGES',
    160: '2-STORY PUD - 1946 & NEWER',
    180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
    190: '2 FAMILY CONVERSION - ALL STYLES AND AGES',
}
# Replace the values with their interpretations
df['MSSubClass'].replace(subclass_dict, inplace=True)
df['MasVnrArea'][df['MasVnrArea'] < 50] = 50
df['MasVnrArea'][(df['MasVnrArea'] > 50) & (df['MasVnrArea'] <200)] = 200
df['MasVnrArea'][(df['MasVnrArea'] > 200) & (df['MasVnrArea'] <500)] = 500
df['MasVnrArea'][df['MasVnrArea'] > 500] = 1600

masvnrarea_dict = {
    50: '50',
    200: '50-200',
    500: '200-500',
    1600: '500+'
}
df['MasVnrArea'].replace(masvnrarea_dict, inplace=True)
df['BsmtUnfSF'][df['BsmtUnfSF'] < 500] = 500
df['BsmtUnfSF'][(df['BsmtUnfSF'] > 500) & (df['BsmtUnfSF'] <1000)] = 1000
df['BsmtUnfSF'][(df['BsmtUnfSF'] > 1000) & (df['BsmtUnfSF'] <1500)] = 1500
df['BsmtUnfSF'][df['BsmtUnfSF'] > 1500] = 2500

bsmtunfsf_dict = {
    500: '500',
    1000: '500-1000',
    1500: '1000-1500',
    2500: '1500+'
}
df['BsmtUnfSF'].replace(bsmtunfsf_dict, inplace=True)
cat_columns = df.loc[:, df.dtypes == object].columns
df = pd.get_dummies(df, columns = cat_columns, drop_first=True)
columns_transform = ['1stFlrSF', 'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'TotalBsmtSF', 'SalePrice']
for col in columns_transform:
    df[col] = np.log(df[col]+1)
    df[col].plot(kind='hist', bins=50, fontsize = 20, figsize= (20, 10), edgecolor='black', linewidth=1.2)
    plt.xlabel(col, fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.show()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
ridge = Ridge()
ridge.fit(X_train, y_train)
preds_train = ridge.predict(X_train)
preds_test = ridge.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
lasso = Lasso()
lasso.fit(X_train, y_train)
preds_train = lasso.predict(X_train)
preds_test = lasso.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
preds_train = rfr.predict(X_train)
preds_test = rfr.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
xg = xgb.XGBRegressor(learning_rate=0.08, n_estimators=1000)
xg.fit(X_train, y_train, eval_metric='rmse')
preds_train = xg.predict(X_train)
preds_test = xg.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
params = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [500, 800, 1000],
}

cv_xg = GridSearchCV(xg, params, cv=5, scoring='neg_mean_squared_error')
cv_xg.fit(X_train, y_train)


print("Tuned parameters {}".format(cv_xg.best_params_))
params = {
    'n_estimators': [10,25],
    'max_depth': [10,50, None],
    'max_features': [5,10,15,20,25],
    'bootstrap': [True, False]
}

cv_rfr = GridSearchCV(rfr, params, cv=10, scoring='neg_mean_squared_error')
cv_rfr.fit(X_train, y_train)


print("Tuned parameters {}".format(cv_rfr.best_params_))
#Getting the importances from grid search
importances = cv_rfr.best_estimator_.feature_importances_

#getting feature names
features = list(X.columns)

feature_importance = sorted(zip(importances, features), reverse=True)

df_importances = pd.DataFrame(feature_importance, columns = ['importances', 'features'])

df_importances.head(30)
rfr_tuned = RandomForestRegressor(n_estimators=50, max_features=125, max_depth=None, min_samples_leaf=3, bootstrap=False)

rfr_tuned.fit(X_train,y_train)
preds_train = rfr_tuned.predict(X_train)
preds_test = rfr_tuned.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))

print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
preds_train = cv_xg.predict(X_train)
preds_test = cv_xg.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))

print("RMSE: %f" % (rmse_train))
print("RMSE: %f" % (rmse_test))
X = scaler.fit_transform(X)
xg.fit(X, y)
cv_xg.fit(X, y)
rfr.fit(X, y)
rfr_tuned.fit(X,y)
ridge.fit(X,y)
lasso.fit(X,y)