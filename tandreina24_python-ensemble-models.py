# packages to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from statistics import mode


#Reading the data sets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#head of fist 5 rows
df_train.head()
#to  get the size of the file
df_train.shape
#all column names
df_train.columns
#list of variables with its statistics
df_train.describe().transpose()
#list of variables with missing values
df_train.columns[df_train.isnull().any()]
#data set only missing variable values
df_train_missing=df_train[df_train.columns[df_train.isnull().any()]]

#Create a new function for missing values:
def num_missing(x):
    return sum(x.isnull())
#Applying per row:
#df_train['nmissing']=df_train_missing.apply(num_missing, axis=1) #axis=1 defines that function is to be applied on each row
#Applying per column:
df_train_missing.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
df_train_missing_table=round(df_train_missing.apply(num_missing, axis=0)/1460,4)*100 #axis=0 defines that function is to be applied on each column
df_train_missing_table =pd.DataFrame({'MissingPercentage':df_train_missing_table})
df_train_missing_table
#None inputation
var1=['Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df_train[var1]=df_train[var1].fillna('NA')

#Mean inputation
df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())

#Mode inputation
df_train['Electrical']=df_train['Electrical'].fillna(mode(df_train['Electrical']))
#list of variables with missing values
df_test.columns[df_test.isnull().any()]
#data set only missing variable values
df_test_missing=df_test[df_test.columns[df_test.isnull().any()]]

#Applying per column:
df_test_missing.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#None inputation
var1=['BsmtQual','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df_test[var1]=df_test[var1].fillna('NA')

#Mean inputation
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())

#Mode inputation
df_test['MSZoning']=df_test['MSZoning'].fillna(mode('MSZoning'))
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(0)
df_test['KitchenQual']=df_test['KitchenQual'].fillna('TA')
df_test['Functional']=df_test['Functional'].fillna(mode('Functional'))
df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(0)
df_test['SaleType']=df_test['SaleType'].fillna(mode('SaleType'))

#check outlier
sns.boxplot(x=df_train['SalePrice']);
#Remove Outlier
df_train=df_train[np.abs(df_train.SalePrice-df_train.SalePrice.mean()) <= (6*df_train.SalePrice.std())]
df_test['TT_SF']=df_test['1stFlrSF']+df_test['2ndFlrSF']+df_test['TotalBsmtSF']+df_test['GarageArea']+df_test['GrLivArea']
df_test['TT_bathrooms']=df_test['FullBath']+df_test['HalfBath']+df_test['BsmtFullBath']+df_test['HalfBath']
df_test['TT_rooms']=df_test['TotRmsAbvGrd']+df_test['BsmtFullBath']+df_test['BsmtHalfBath']

df_train['TT_SF']=df_train['1stFlrSF']+df_train['2ndFlrSF']+df_train['TotalBsmtSF']+df_train['GarageArea']+df_train['GrLivArea']
df_train['TT_bathrooms']=df_train['FullBath']+df_train['HalfBath']+df_train['BsmtFullBath']+df_train['HalfBath']
df_train['TT_rooms']=df_train['TotRmsAbvGrd']+df_train['BsmtFullBath']+df_train['BsmtHalfBath']

#delete the ID variable in train data
#del df_train['Id']
#histogram
sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#histogram
sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#List of numeric variables
numeric_vars = df_train.dtypes[df_train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_vars = df_train[numeric_vars].skew().sort_values(ascending=False)
skewness = pd.DataFrame({'Skew':skewed_vars})

#keeping only the one with high skewness
skewness = skewness[abs(skewness['Skew']) > 0.7]
skewness

#Transformation 
skewed_vars = skewness.index
df_train[skewed_vars] = np.log(df_train[skewed_vars]+0.001)
df_test[skewed_vars] = np.log(df_test[skewed_vars]+0.001)
#convert categorical variable into dummy
df_train=pd.get_dummies(df_train)
df_test=pd.get_dummies(df_test)

#Sort column to get the same order 
final_train, final_test = df_train.align(df_test,join='inner',axis=1)

#correlation
corr_df=df_train.corr()

#list of variables with high correaltion with Sale price
high_corr_df=corr_df[corr_df['SalePrice']>0.5]
high_corr_df=high_corr_df[list(corr_df[corr_df['SalePrice']>0.5].index)]

#graph
heatmap_df=high_corr_df
plt.subplots(figsize=(10,8))
sns.heatmap(heatmap_df,annot=True);
#Distibution and correation graph of the variables with highest correaltion
sns.pairplot(df_train[['SalePrice','OverallQual','GarageCars','GrLivArea', 'GarageArea']]);
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# GradientBoostingRegressor:
# define pipeline
GBR_model = make_pipeline(GradientBoostingRegressor())
# cross validation score
score = cross_val_score(GBR_model,final_train, df_train.SalePrice, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))
# fit and make predictions
GBR_model.fit(final_train,df_train.SalePrice)
predictions= GBR_model.predict(final_test)
# Import libaries for  models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Lasso

# Metrics for root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt
# Initialize models
lr = LinearRegression()
rd = Ridge()
rf = RandomForestRegressor(
    n_estimators = 12,
    max_depth = 3,
    n_jobs = -1)
gb = GradientBoostingRegressor()
lasso=Lasso(alpha =0.0005, random_state=1)

#nn = MLPRegressor(
#    hidden_layer_sizes = (90, 90),
#    alpha = 2.75)
# Initialize Ensemble
model = StackingRegressor(
    regressors=[rf, gb, rd, lasso],
    meta_regressor=lr)

# Fit the model on our data
model.fit(final_train, df_train.SalePrice);
# Predict training data
y_pred = model.predict(final_train)
print(sqrt(mean_squared_error(df_train.SalePrice, y_pred)))
# Predict test data
y_pred = model.predict(final_test)
# Fit the model on our data
lasso.fit(final_train, df_train.SalePrice)

score = cross_val_score(lasso,final_train, df_train.SalePrice, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.exp(predictions)})
submission.to_csv('submissionGB.csv', index=False)
submission.head()
# Create empty submission dataframe
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.exp(y_pred)})

# Output submission file
submission.to_csv('submissionEM.csv',index=False)
submission.head()