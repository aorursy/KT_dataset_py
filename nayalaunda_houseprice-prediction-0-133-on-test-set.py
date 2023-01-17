from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import missing_data_imputers as mdi
from feature_engine import discretisers as dsc
from feature_engine import categorical_encoders as ce
import seaborn as sns
from feature_engine import variable_transformers as vt
import scipy.stats as stats
pd.pandas.set_option('display.max_columns', None)
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from feature_engine.outlier_removers import Winsorizer
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
##LOAD DATASET
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
datas = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
data.head()
# let's inspect the type of variables in pandas
for var in data.columns:
    print(var,data[var].dtypes)
#find categorical variables
categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))
data[categorical].head()
# make a list of the numerical variables 
numerical = [var for var in data.columns if data[var].dtype!='O']
data[numerical].head()
# list of variables that contain year information

year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]

year_vars
data[year_vars].head()
# plot median house price per month in which it was sold
data.groupby('MoSold')['SalePrice'].median().plot()
plt.title('House price variation along the year')
plt.ylabel('median House price')
# let's visualise the values of the discrete variables
discrete = []

for var in numerical:
    if len(data[var].unique()) < 20 and var not in year_vars:
        print(var, ' values: ', data[var].unique())
        discrete.append(var)
print()
print('There are {} discrete variables'.format(len(discrete)))
data[discrete].head()
# find continuous variables
numerical = [var for var in numerical if var not in discrete and var not in [
    'Id', 'SalePrice'] and var not in year_vars]

print('There are {} numerical and continuous variables'.format(len(numerical)))
data[numerical].head()
#imputing missing variables for categorical variables
data[categorical].isnull().sum()
#impute Missing values with an indicator "M"
for var in categorical:
    data[var] = np.where(data[var].isnull(),'M',data[var])
    datas[var] = np.where(datas[var].isnull(),'M',datas[var])
data[categorical].isnull().sum()
#Missing values replaced.
#now lets check for missing values in discrete variables

data[discrete].isnull().sum()

####NOTE:::replace here
for var in discrete:
    datas[var] = np.where(datas[var].isnull(),datas[var].mode(),datas[var])
datas[discrete].isnull().sum()
#lets check for missing values in numerical varibels now
data[numerical].isnull().sum()
'''LotFrontage: Linear feet of street connected to property , 
thus we can assume there are missing values as these houses have no conectivity to street area
we will impute these missing values with arbitary value -1 to mark this observation'''
data['LotFrontage'] = np.where(data['LotFrontage'].isnull(),-1,data['LotFrontage'])
datas['LotFrontage'] = np.where(datas['LotFrontage'].isnull(),-1,datas['LotFrontage'])
data['MasVnrArea'] = np.where(data['MasVnrArea'].isnull(),data['MasVnrArea'].mean(),data['MasVnrArea'])
datas['MasVnrArea'] = np.where(datas['MasVnrArea'].isnull(),datas['MasVnrArea'].mean(),datas['MasVnrArea'])
datas['BsmtFinSF1'] = np.where(datas['BsmtFinSF1'].isnull(),datas['BsmtFinSF1'].mean(),datas['BsmtFinSF1'])
datas['BsmtFinSF2'] = np.where(datas['BsmtFinSF2'].isnull(),datas['BsmtFinSF2'].mean(),datas['BsmtFinSF2'])
datas['BsmtUnfSF'] = np.where(datas['BsmtUnfSF'].isnull(),datas['BsmtUnfSF'].mean(),datas['BsmtUnfSF'])
datas['TotalBsmtSF'] = np.where(datas['TotalBsmtSF'].isnull(),datas['TotalBsmtSF'].mean(),datas['TotalBsmtSF'])
datas['GarageArea'] = np.where(datas['GarageArea'].isnull(),datas['GarageArea'].mean(),datas['GarageArea'])
data[numerical].isnull().sum()
#Missing values for numerical features imputed.
datas[numerical].isnull().sum()
####NOTE:::do mean median imputing for dataset...
#one hot categorical encoding for categorical variables.
y_df = data['SalePrice']
#re-casting discrete variables to categorical
data[discrete] =data[discrete].astype('O')
datas[discrete] =datas[discrete].astype('O')

#rare-label encoding
del data['SalePrice']
rareenc = ce.RareLabelCategoricalEncoder(
        tol=0.05, n_categories=6, variables=categorical+discrete)
rareenc.fit(data)
data = rareenc.transform(data)
datas = rareenc.transform(datas)
##space for datas
enc = OneHotCategoricalEncoder(
    top_categories=None,
    variables=categorical, # we can select which variables to encode
    drop_last=True) 
enc.fit(data,y_df)
data= enc.transform(data)
datas = enc.transform(datas)
datas
#ordinal encoding for discrete variables
orden = ce.OrdinalCategoricalEncoder(
        encoding_method='ordered', variables=discrete)
orden.fit(data,y_df)
data = orden.transform(data)
datas = orden.transform(datas)
##space for datas
datas[discrete].isnull().sum()
datas['FullBath'] = np.where(datas['FullBath'].isnull(),datas['FullBath'].mode(),datas['FullBath'])
datas['Fireplaces'] = np.where(datas['Fireplaces'].isnull(),datas['Fireplaces'].mode(),datas['Fireplaces'])
datas['GarageCars'] = np.where(datas['GarageCars'].isnull(),datas['GarageCars'].mode(),datas['GarageCars'])
datas[discrete].isnull().sum()
#now we will assess normality for numerical continuous variables
def diagnostic_plots(df, variable):
    
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()

for var in numerical:
    print(var)
    diagnostic_plots(data, var)
#transforming variables to see best function
from feature_engine import variable_transformers as vt
transform = ['LotArea','BsmtUnfSF','GrLivArea','GarageArea']
lt = vt.YeoJohnsonTransformer(variables = transform)
lt.fit(data)
data = lt.transform(data)
datas = lt.transform(datas)
#I selected variables for transformation by selecting individual graphs
#Outlier enginerring
windsoriser = Winsorizer(distribution='quantiles', # choose from skewed, gaussian or quantiles
                          tail='both', # cap left, right or both tails 
                          fold=0.05,
                          variables=numerical)
windsoriser.fit(data)
data_tf = windsoriser.transform(data)
datas = windsoriser.transform(datas)
data = data_tf
#engineering temporal variables
def elapsed_years(df, var):
    # capture difference between year variable and
    # year the house was sold
    
    df[var] = df['YrSold'] - df[var]
    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    data = elapsed_years(data, var)
    datas = elapsed_years(datas, var)
data[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
# drop YrSold
data.drop('YrSold', axis=1, inplace=True)
datas.drop('YrSold', axis=1, inplace=True)
del data['Id']
del datas['Id']
#feature Selection
#removing constant features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0)
sel.fit(data)  # fit finds the features with zero variance

sum(sel.get_support())
data= sel.transform(data)
datas = sel.transform(datas)

data.shape , datas.shape
datas
data = pd.DataFrame(data)
datas = pd.DataFrame(datas)
#removing co-related features
data
#del data[32]
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = correlation(data, 0.8)
len(set(corr_features))
data.drop(labels=corr_features, axis=1, inplace=True)
datas.drop(labels=corr_features, axis=1, inplace=True)

data.shape #, X_test.shape
#datas.drop(labels=corr_features, axis=1, inplace=True)
datas.shape
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    y_df,
                                                    test_size=0.1,
                                                    random_state=0)

X_train.shape, X_test.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(X_train)
x_test = sc.transform(X_test)
x_test2 = sc.transform(datas)
#Modelling
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb=xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.04, 
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=6, n_estimators=2000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.7, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(x, y_train)
xgb_train_pred = model_xgb.predict(x)
xgb_pred = np.expm1(model_xgb.predict(x_test))
print(rmsle(y_train, xgb_train_pred))

xgb_pred2 = model_xgb.predict(x_test2)
mean_squared_error(y_train, xgb_train_pred)

