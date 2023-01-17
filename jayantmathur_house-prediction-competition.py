! pip install jcopml
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
import matplotlib.pyplot as plt

import seaborn as sns 
from scipy.stats import norm

df_train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#save ID
train_id = df_train['Id']
test_id = df_test['Id']

#drop ID
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)

print(df_train.shape)
print(df_test.shape)
house_data=pd.concat([df_train,df_test],axis=0)
house_data = house_data.reset_index(drop=True)
house_data.tail(4)
house_data.info()
def show_missing_info(house_data):
    missing_info = house_data.isna().sum().reset_index(drop=False)
    missing_info.columns = ["column","rows"]
    missing_info["missing_percent"] = (missing_info["rows"]/house_data.shape[0])*100
    missing_info = missing_info[missing_info["rows"]>0].sort_values(by="missing_percent",ascending=False)
    return missing_info
missing_df = show_missing_info(house_data)
missing_df
delete_rows_cols = missing_df[missing_df["rows"]<20]["column"].tolist()
delete_rows_cols
house_data.dropna(axis=0,how="any",subset=delete_rows_cols,inplace=True)
print(house_data.shape)
df_train["SalePrice"].describe()
sns.distplot(df_train['SalePrice'], fit=norm);
corrmat = df_train.corr(method='pearson')
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmat , square=True,annot=True,cmap='RdYlBu');
f, ax = plt.subplots(figsize=(10, 10))
k = 11 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', square=True, annot_kws={'size': 12}, 
                 yticklabels=cols.values, xticklabels=cols.values, cmap='RdYlBu')
plt.show()
sns.set(style='whitegrid')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
sns.pairplot(df_train[cols], height = 2.5)
plt.show()
sns.set(style='whitegrid')
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice']);

axes2 = fig.add_axes([1.1, 0.1, 0.8, 0.8])
sns.scatterplot(x=df_train['TotalBsmtSF'], y=df_train['SalePrice']);
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train.drop(df_train.index[1298], inplace=True)
df_train.drop(df_train.index[523], inplace=True)

df_train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1]
df_train.drop(df_train.index[1298], inplace=True)
sns.set(style='whitegrid')
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice'],color='g');

axes2 = fig.add_axes([1.1, 0.1, 0.8, 0.8])
sns.scatterplot(x=df_train['TotalBsmtSF'], y=df_train['SalePrice'],color='g');
def input(df):
    col_name = df.columns
    for col_name in df:
        
        df["PoolQC"] = df["PoolQC"].fillna("None")
        df["MiscFeature"] = df["MiscFeature"].fillna("None")
        df["Alley"] = df["Alley"].fillna("None")
        df["Fence"] = df["Fence"].fillna("None")
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
                        df[col] = df[col].fillna('None')

        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
                        df[col] = df[col].fillna(0)

        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            df[col] = df[col].fillna(0)

        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            df[col] = df[col].fillna('None')

        df["MasVnrType"] = df["MasVnrType"].fillna("None")
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        df['MSZoning'] = df['MSZoning'].fillna('RL')

        df["Functional"] = df["Functional"].fillna('Typ')

        df['Electrical'] = df['Electrical'].fillna('SBrkr')

        df['KitchenQual'] = df['KitchenQual'].fillna('TA')

        df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
        df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')

        df['SaleType'] = df['SaleType'].fillna('WD')

        df['MSSubClass'] = df['MSSubClass'].fillna("None")

        df.drop(columns=['Utilities','1stFlrSF','GarageYrBlt','TotRmsAbvGrd','GarageArea'],inplace=True)
        return df
df_train=input(df_train)
df_train.info()
print('df_train shape = {}'.format(df_train.shape))
df_train.head()
df_test=input(df_test)
df_test.shape
sns.distplot(df_train['SalePrice'], fit=norm);
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe
target_transformer = ColumnTransformer([('target', num_pipe(scaling='standard',transform='box-cox'),['SalePrice'])])
y_target = target_transformer.fit_transform(df_train)
y_target = pd.Series(y_target.flatten())
sns.distplot(y_target, fit=norm);
num = df_train.select_dtypes(exclude=['object'])
cat = df_train.select_dtypes(include=['object'])
# Ordinal
cat_or = cat[['Street', 'Alley', 'LandContour', 'LandSlope', 'ExterQual', 'ExterCond',
            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'HeatingQC','CentralAir', 'KitchenQual', 'Functional',
            'FireplaceQu','GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
            'PoolQC', 'Fence', 'SaleCondition' ]]

# Nominal
cat_nom = cat[['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood','Condition1', 'Condition2',
             'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
             'MasVnrType', 'Heating', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType']]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
X = df_train.drop(columns='SalePrice')
y = y_target #target yang sudah di scaling dan transform

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson'))   
])

cat_ord_pipe = Pipeline([
    ('encoder', OrdinalEncoder())
])

cat_nom_pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('numeric1', num_pipe,  [
       'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']),
    
    ('categoric1', cat_ord_pipe , [
       'Street', 'Alley', 'LandContour', 'LandSlope', 'ExterQual', 'BsmtQual', 
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','CentralAir', 'KitchenQual',
       'FireplaceQu','GarageFinish', 'GarageQual', 'PavedDrive', 'Fence', 'SaleCondition' ]),
    
    ('categoric2', cat_nom_pipe, [
       'MSZoning', 'LotShape', 'LotConfig', 'Neighborhood','Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'Heating', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType',
       'Foundation', 'HeatingQC','Functional','ExterCond','BsmtCond','GarageCond', 'PoolQC'])    
])
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))
])

parameter = {'algo__n_estimators': np.arange(100,200),
 'algo__max_depth': np.arange(20,80),
 'algo__max_features': np.arange(0.1,1),
 'algo__min_samples_leaf': np.arange(1,20)}

model = RandomizedSearchCV(pipeline, parameter, cv=3, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
from sklearn.preprocessing import LabelEncoder 
# df.apply(LabelEncoder().fit_transform)
# le = LabelEncoder() 
# filled=fil.le.fit_transform
def dummyEncode(train):
    columnsToEncode = list(df_train.select_dtypes(include=['object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df_train[feature] = le.fit_transform(df_train[feature])
        except:
            print('Error encoding '+feature)
    return train

fill = dummyEncode(df_train)

fill.info()
corrmat1 = fill.corr(method='pearson')
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrmat1)
f, ax = plt.subplots(figsize=(10, 10))
k = 11 
cols = corrmat1.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(fill[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', square=True, annot_kws={'size': 12}, 
                 yticklabels=cols.values, xticklabels=cols.values, cmap='RdYlBu')
plt.show()
features= fill[[ 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
       'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
       'BsmtFinSF1']]
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
X=features
calc_vif(X)
feat1=features.drop(["YearBuilt",'YearRemodAdd'],axis=1)
X=feat1
calc_vif(X)
feat2=feat1.drop(['OverallQual'],axis=1)
X_=feat2
calc_vif(X)
Y=df_train['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X_,Y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)
coeff_df = pd.DataFrame(regressor.coef_.T, X_.columns, columns=['Coefficient'])  
coeff_df
y_pred = regressor.predict(X_test)
y_pred=pd.DataFrame(data=y_pred)
y_pred
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred))*100)
regressor.score(X_test,Y_test)
from sklearn.svm import SVR
regressor_sr = SVR(kernel='linear',C=1.0,degree=6)
regressor_sr.fit(X_train,Y_train)
from sklearn import metrics
y_pred_sr = regressor_sr.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",regressor_sr.score(X_test,Y_test))
from sklearn.tree import DecisionTreeRegressor
regressor_dr = DecisionTreeRegressor(random_state = 42,max_depth=7)  
  
# fit the regressor with X and Y data 
regressor_dr.fit(X_train, Y_train)
y_pred_dr = regressor_dr.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",regressor_dr.score(X_test,Y_test))
from sklearn import ensemble 
reg = ensemble.GradientBoostingRegressor()
reg.fit(X_train, Y_train)
preds=reg.predict(X_test)
reg.score(X_test,Y_test)
from xgboost import XGBRegressor 
xg_reg = XGBRegressor(objective ='reg:squarederror',booster='gbtree',random_state=42,learning_rate = 0.1, alpha = 10, n_estimators = 100)
xg_reg.fit(X_train,Y_train)
preds = xg_reg.predict(X_test)
xg_reg.score(X_test,Y_test)