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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',500)
train.head()
test.head()
train.shape,test.shape
data = pd.concat([train,test])
data.shape
data.isnull().sum()
numeric_columns = [i for i in data.columns if data[i].dtypes != 'object']
categorical_columns = [i for i in data.columns if data[i].dtypes == 'object']
print('NUMERIC COLUMNS')
print(numeric_columns)
print('*'*50)
print('CATEGORICAL COLUMNS')
print(categorical_columns)
numeric_and_nan = [i for i in data.columns if data[i].dtypes != 'object' and data[i].isnull().sum() > 0]
categorical_and_nan = [i for i in data.columns if data[i].dtypes == 'object' and data[i].isnull().sum() > 0]
data[numeric_columns].isnull().sum()
data[numeric_and_nan].isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.style.use("ggplot")
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
def plot_kdeplot(data,column,hue=None,multiple=None,cumulative=False, common_norm=False, common_grid=False,log_scale=False):
    return sns.kdeplot(x=data[column],hue=hue,multiple=multiple,cumulative=cumulative,common_norm=common_norm,common_grid=common_grid,log_scale=log_scale)
    
def plot_distplot(df,column,color='red'):
    return sns.distplot(df[column],color=color)
def get_nan_index(df,column):
    return df.loc[pd.isna(df[column]),:].index
index_ = []
for feature in numeric_and_nan:
    index_.append(get_nan_index(data,feature))
for i in index_:
    plt.hist(i, density=True, bins=20)
    plt.grid()
    plt.show()
def cal_outliers_quantile(df,column):
    IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
    lower_bridge = df[column].quantile(0.25) - (IQR*1.5)
    upper_bridge=df[column].quantile(0.75) + (IQR*1.5)
    
    return lower_bridge,upper_bridge
for feature in numeric_columns:
    lower_lot_frontage,upper_lot_frontage = cal_outliers_quantile(data,feature)
    print('Outliers for',feature,'is',lower_lot_frontage,',',upper_lot_frontage)
    print(data[(data[feature] < lower_lot_frontage) | (data[feature] > upper_lot_frontage)].shape[0])
    print('*'*50)
plot_distplot(data,'LotFrontage')
data['newLotFrontage_median'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
plot_distplot(data,'newLotFrontage_median')
data['newLotFrontage_mean'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
plot_distplot(data,'newLotFrontage_mean')
data['LotFrontage'].head()
sns.kdeplot(data['LotFrontage'])
data[data['LotFrontage'] > 150].shape
#data.drop(data[data['LotFrontage'] >= 150].index,inplace=True)
plot_distplot(data,'LotFrontage')

sns.kdeplot(data['LotFrontage'])
from sklearn.impute import KNNImputer

imputer =  KNNImputer(n_neighbors=3)
sns.distplot(data['MasVnrArea'])
print("Number of house which don't have Masonry veneer wall:",data[data['MasVnrArea'] <= 0].shape[0])
plt.hist(data['MasVnrArea'])
plt.show()
data[data['MasVnrArea'] > 1000].shape
#data.drop(data[data['MasVnrArea'] > 1000].index,inplace=True)
data['MasVnrArea'].fillna(0.0,inplace=True)
data['BsmtFinSF1'].head()
sns.distplot(data['BsmtFinSF1'])
#data.drop(data[data['BsmtFinSF1'] > 2000].index,inplace=True)
data['BsmtFinSF1'].bfill(inplace=True)
data['BsmtFinSF2'].bfill(inplace=True)
data['BsmtFinSF2'].unique()
plt.hist(data['BsmtFinSF2'])
plt.show()
data['BsmtUnfSF'].unique()
relation_bsmt = data[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].corr()
relation_bsmt
data.drop(['BsmtFinSF2','BsmtUnfSF'],axis=1,inplace=True)
data['TotalBsmtSF'].bfill(inplace=True)
numeric_and_nan
data['BsmtFullBath'].bfill(inplace=True)
data['BsmtHalfBath'].bfill(inplace=True)
sns.distplot(data['GarageYrBlt'])
data[['GarageCars','GarageArea']].corr()
#dropping garagecars
data.drop('GarageCars',axis=1,inplace=True)
data.shape
data[categorical_and_nan].isnull().sum()
for value in categorical_and_nan:
    print('Value:',value)
    print(data[value].value_counts())
    print('*'*50)
data['Fence'].fillna('None',inplace=True)
data['FireplaceQu'].fillna('None',inplace=True)
data.drop(['PoolQC','MiscFeature','Utilities'],axis=1,inplace=True)
data['MSZoning'].ffill(inplace=True)
data.drop('Alley',axis=1,inplace=True)
categorical_and_nan.remove('PoolQC')
categorical_and_nan.remove('MiscFeature')
categorical_and_nan.remove('Utilities')
categorical_and_nan.remove('Alley')
categorical_and_nan.remove('Fence')
categorical_and_nan.remove('FireplaceQu')
def encode_categorical(feature_list):
    for label in feature_list:
        temp_names = list(data[label].unique())
        temp_values = list(range(0,len(temp_names)))
        temp_dict = dict(zip(temp_names,temp_values))
        data[label] = data[label].map(temp_dict)
        del temp_names,temp_values,temp_dict
    return data    
new_categorical_features = [i for i in data.columns if data[i].dtypes == 'object']
data = encode_categorical(categorical_and_nan)
data.isnull().sum()
data.drop(['newLotFrontage_median','newLotFrontage_mean'],axis=1,inplace=True)
data['GarageYrBlt'].fillna(2000,inplace=True)
data['GarageArea'].ffill(inplace=True)
data.head()
data.isnull().sum()
for label in new_categorical_features:
    print('Label',label)
    print(data[label].value_counts())
    print('*'*50)
data.drop(['Street','Condition2','RoofMatl'],axis=1,inplace=True)
new_categorical_features.remove('Street')
new_categorical_features.remove('Condition2')
new_categorical_features.remove('RoofMatl')
data = encode_categorical(new_categorical_features)
data.head()
data.dtypes == 'object'
from statsmodels.graphics.gofplots import qqplot
fig = qqplot(train['SalePrice'],line='s')
fig.show()
data.head()
data_corr = data.corr()
data[['MSSubClass','MSZoning']].corr()
data[['LotFrontage','LotArea','LotShape','LotConfig','SalePrice']].corr()
data.drop('LotArea',axis=1,inplace=True)
data[['Condition1','SalePrice']].corr()
data.drop(['Condition1'],axis=1,inplace=True)
data[['OverallQual','OverallCond','SalePrice']].corr()
data[['Exterior1st','Exterior2nd','SalePrice']].corr()
data.drop('Exterior1st',axis=1,inplace=True)
data[['MasVnrType','MasVnrArea','SalePrice']].corr()
bsmt_list = data.columns[24:31]
bsmt_list
data[bsmt_list].corr()
data[['Heating','HeatingQC']].corr()
data[['1stFlrSF','2ndFlrSF']].corr()
data.columns
data[['GarageType','GarageYrBlt','GarageFinish','GarageArea','GarageQual','GarageCond','SalePrice']].corr()
data.drop(['GarageQual','GarageCond'],axis=1,inplace=True)
data.shape
sorted_data_corr = data_corr['SalePrice'].sort_values(ascending=False)
type(sorted_data_corr)
sorted_data_corr.index[0]
sorted_data_corr

dictionary_corr = {}
for i in range(len(sorted_data_corr.index)):
    if sorted_data_corr[i] >= 0.2 or sorted_data_corr[i] <=  -0.2:
        dictionary_corr[sorted_data_corr.index[i]] = sorted_data_corr[i]
len(dictionary_corr),len(sorted_data_corr)
dictionary_corr
for i in data.columns:
    if i not in dictionary_corr.keys():
        data.drop(i,axis=1,inplace=True)
set(data.columns).difference(dictionary_corr.keys())
data.head()
for feature in data.columns:
    lower_lot_frontage,upper_lot_frontage = cal_outliers_quantile(data,feature)
    print('Outliers for',feature,'is',lower_lot_frontage,',',upper_lot_frontage)
    print(data[(data[feature] < lower_lot_frontage) | (data[feature] > upper_lot_frontage)].shape[0])
    print('*'*50)
data.columns
for i in data.columns:
    print(i)
    print(data[i].unique())
    print("*"*50)
sns.kdeplot(data['LotFrontage'])
def calculate_quantiles(df,column):
    for i in range(0,100,10):
        var =df[column].values
        var = np.sort(var,axis = None)
        print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
    print("99th percentle value is",df[column].quantile(0.99))    
    print ("100 percentile value is ",df[column].quantile(1))
for i in data.columns:
    if len(data[i].unique()) > 30:
        print(i,'..........')
        calculate_quantiles(data,i)
        print("Number of data points greater than 99th percentile is",data[data[i] > data[i].quantile(0.99)].shape[0])
        print("Index of data points greater than 99th percentile is",data[data[i] > data[i].quantile(0.99)].index)
        print("*"*50)
sns.distplot(data['LotFrontage'])
data.isnull().sum()
for i in data.columns:
    if len(data[i].unique()) > 30 and i != 'SalePrice':
        print(i,'..........')
        median = data.loc[data[i] < data[i].quantile(0.99), i].median()
        data.loc[data[i] > data[i].quantile(0.99), i] = np.nan
        data[i].fillna(median,inplace=True)
        print('Done')
        print("*"*50)
sns.distplot(data['LotFrontage'],fit=stats.norm)
for i in data.columns:
    if data[i].skew() > 1.0:
        print(i,':',data[i].skew())
for i in data.columns:
    if data[i].skew() > 1.0:
        print(i,':',len(data[i].unique()))
import pylab
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stats.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()
print('for training data')
plot_data(train,'MasVnrArea')
plot_data(data,'MasVnrArea')
plot_data(train,'WoodDeckSF')
print('For whole data')
plot_data(data,'OpenPorchSF')
print('For training data')
plot_data(train,'OpenPorchSF')
plot_data(train,'SalePrice')
train.shape
X_train_data = data.iloc[:1460,:]
X_test_data = data.iloc[1460:,:]
X_train_data['SalePrice'].isnull().sum()
X_train_data['SalePrice'],params = stats.boxcox(X_train_data['SalePrice'])
plot_data(X_train_data,'SalePrice')
for i in data.columns:
    if data[i].skew() > 1.0:
        print(i,':',len(data[i].unique()))
print('MasVnrArea',X_train_data[X_train_data['MasVnrArea'] <= 0].shape[0])
print('WoodDeckSF',X_train_data[X_train_data['WoodDeckSF'] <= 0].shape[0])
print('OpenPorchSF',X_train_data[X_train_data['OpenPorchSF'] <= 0].shape[0])
print('MasVnrArea',X_train_data[X_train_data['MasVnrArea'] < 0].shape[0])
print('WoodDeckSF',X_train_data[X_train_data['WoodDeckSF'] < 0].shape[0])
print('OpenPorchSF',X_train_data[X_train_data['OpenPorchSF'] < 0].shape[0])
X_train_data['MasVnrArea_boxcox'],params = stats.boxcox(X_train_data['MasVnrArea'] + 1)
X_test_data['MasVnrArea_boxcox'],params = stats.boxcox(X_test_data['MasVnrArea'] + 1)

X_train_data['WoodDeckSF_boxcox'],params = stats.boxcox(X_train_data['WoodDeckSF'] + 1)
X_test_data['WoodDeckSF_boxcox'],params = stats.boxcox(X_test_data['WoodDeckSF'] + 1)

X_train_data['OpenPorchSF_boxcox'],params = stats.boxcox(X_train_data['OpenPorchSF'] + 1)
X_test_data['OpenPorchSF_boxcox'],params = stats.boxcox(X_test_data['OpenPorchSF'] + 1)
X_test_data.drop('SalePrice',axis=1,inplace=True)
drop_final_cols = []
for i in X_train_data.columns:
    if X_train_data[i].skew() > 1.0 and len(X_train_data[i].unique()) > 30:
        print(i,':',X_train_data[i].skew())
        drop_final_cols.append(i)
for i in X_test_data.columns:
    if X_test_data[i].skew() > 1.0 and len(X_test_data[i].unique()) > 30:
        print(i,':',X_test_data[i].skew())
X_train_data.drop(drop_final_cols,axis=1,inplace=True)
X_test_data.drop(drop_final_cols,axis=1,inplace=True)
X_train_data.shape,train.shape
X_test_data.shape,test.shape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.model_selection import KFold,cross_val_score
from sklearn import metrics
X_train = X_train_data.drop('SalePrice',axis=1)
y_train = X_train_data['SalePrice']
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test_data = scalar.transform(X_test_data)
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
model_xgb = xgboost.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(model_xgb)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmsle(y, y_pred):
    return np.sqrt(metrics.mean_squared_error(y, y_pred))
model_xgb.fit(X_train, y_train)
xgb_train_pred = model_xgb.predict(X_train)
xgb_pred = np.expm1(model_xgb.predict(X_test_data))
print(rmsle(y_train, xgb_train_pred))
test_ID = test['Id']
xgb_pred.shape,test.shape,X_test_data.shape
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = xgb_pred
sub.to_csv('submission_3.csv',index=False)

hello
y_train.head()
sub['SalePrice'].head()
sns.kdeplot(y_train,label='train')
sns.kdeplot(sub['SalePrice'],label='sub')
plt.legend()
submission.head()
submission['Boxcox'],_ = stats.boxcox(submission['SalePrice'])
submission['Boxcox'].head()
sns.kdeplot(sub['SalePrice'],label='submission')
sns.kdeplot(submission['Boxcox'],label='boxcox')
sns.kdeplot(sub['SalePrice'],label='submission')
sns.kdeplot(submission['Boxcox'],label='boxcox')
plt.legend()
