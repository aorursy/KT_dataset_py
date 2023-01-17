import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm, skew #for some statistics
properties = pd.read_csv("../input/DC_Properties.csv",index_col="Unnamed: 0")
properties.head()
properties.describe()
properties.info()
properties.shape
total = properties.isnull().sum().sort_values(ascending=False)
total[total>0]
percent = (properties.isnull().sum()/properties.isnull().count()).sort_values(ascending = False)
#print(percent)
pd1 = pd.concat([total,percent],axis =1 ,keys=['Total','Percent'])
pd2 = pd1[pd1['Total']>0]
print(pd2)
#Removing columns with no impact on Price
properties.dropna(subset=['AYB','Y','QUADRANT','X','ASSESSMENT_NBHD','WARD','ZIPCODE','LATITUDE','LONGITUDE','CENSUS_TRACT'], inplace=True)
#fILLING NA Values
mean_livingGBA = np.mean(properties.LIVING_GBA)
properties['LIVING_GBA']=properties.LIVING_GBA.fillna(mean_livingGBA)
mean_GBA = np.mean(properties.GBA)
properties['GBA']=properties.GBA.fillna(mean_GBA)
properties['SALEDATE'] = pd.to_datetime(properties['SALEDATE'])
properties.dropna(subset=['SALEDATE'], inplace=True)
#GIS_LAST_MOD_DTTM
properties['GIS_LAST_MOD_DTTM'] = pd.to_datetime(properties['GIS_LAST_MOD_DTTM'])
properties.dropna(subset=['GIS_LAST_MOD_DTTM'], inplace=True)
properties["ZIPCODE"] = properties.ZIPCODE.astype(object)
properties["ASSESSMENT_SUBNBHD"] = properties.ASSESSMENT_SUBNBHD.astype(object)

properties["NUM_UNITS"] = properties["NUM_UNITS"].fillna(0)
properties["STORIES"] = properties["STORIES"].fillna(0)
properties["KITCHENS"] = properties["KITCHENS"].fillna(0)
properties["GRADE"] = properties["GRADE"].fillna("None")
properties["STRUCT"] = properties["STRUCT"].fillna("None")
properties["ROOF"] = properties["ROOF"].fillna("None")
properties["STYLE"] = properties["STYLE"].fillna("None")
properties["CNDTN"] = properties["CNDTN"].fillna("None")
#removing columns with no impact on price
properties = properties.drop(['CMPLX_NUM', 'YR_RMDL','FULLADDRESS','NATIONALGRID','CENSUS_BLOCK',
                             'STATE','CITY','EXTWALL','INTWALL'],axis=1)

properties = properties.drop_duplicates()
unknown = properties[properties['PRICE'].isnull()]
unknown.head()
#removing unknown data
properties.dropna(subset=['PRICE'], inplace=True)
int_cols = [key for key in dict(properties.dtypes) if dict(properties.dtypes)[key] in ['int64','float64']]
int_cols
def drop_outliers(df, field_name):
    distance = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
    df.drop(df[df[field_name] > distance + np.percentile(df[field_name], 75)].index, inplace=True)
    df.drop(df[df[field_name] < np.percentile(df[field_name], 25) - distance].index, inplace=True)
#Removing outliers
for i in int_cols:
    drop_outliers(properties,i)
sns.boxplot(y=properties.PRICE)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
cols =('QUADRANT','SOURCE','AC','HEAT','WARD','ASSESSMENT_NBHD','QUALIFIED','ZIPCODE','ASSESSMENT_SUBNBHD','STYLE','STRUCT','GRADE','CNDTN','ROOF')
#cols = ('HEAT','AC','QUALIFIED','STYLE','STRUCT','GRADE','CNDTN','EXTWALL','ROOF','INTWALL','SOURCE','WARD','QUADRANT')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(properties[c].values)) 
    properties[c] = lbl.transform(list(properties[c].values))

# shape        
print('Shape all_data: {}'.format(properties.shape))
properties.info()
#Check the new distribution 
sns.distplot(properties['PRICE'] , fit=norm)
#Converting price to normal distribution using log transformation
properties["PRICE"] = np.log1p(properties["PRICE"])
#Check the new distribution 
sns.distplot(properties['PRICE'] , fit=norm)
properties.corr(method='pearson')>0.7
# Removing columns with High auto co-relation

del properties['X']
del properties['Y']
del properties['BEDRM']

del properties['WARD']
del properties['ASSESSMENT_SUBNBHD']
del properties['SOURCE']
del properties['SALEDATE']
del properties['GIS_LAST_MOD_DTTM']

# Splitting into train and test data
from sklearn.cross_validation import train_test_split

train,test = train_test_split(properties, train_size=0.8 , random_state=100)
import matplotlib.pyplot as plt
import seaborn as sns
total = train.isnull().sum().sort_values(ascending=False)
total[total>0]
df1 = train.copy()
# First extract the target variable which is our House prices
Y = df1.PRICE.values
# Drop price from the house dataframe and create a matrix out of the house data
X = df1.drop(['PRICE'], axis=1)
#X = df1.as_matrix()
# Store the column/feature names into a list "colnames"
int_cols = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] in ['float64', 'int64','uint8']]
X_train = train[int_cols]
y_train = train['PRICE']
X_test = test[int_cols]
y_test = test['PRICE']
print(int_cols)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import matplotlib.pyplot as plt
sns.distplot(train['PRICE'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['PRICE'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['PRICE'], plot=plt)
plt.show()
import statsmodels.formula.api as sm

model = sm.OLS(y_train,X_train)
fit = model.fit()

print(fit.summary())

print(mean_squared_error(fit.predict(X_test), y_test))
print(np.round(r2_score(y_test, fit.predict(X_test))*100,2),'%')

# Plot the coefficients
#plt.plot(range(len(int_cols)), fit.coef_)
#plt.xticks(range(len(int_cols)), int_cols, rotation=60) 
#plt.margins(0.02)
#plt.show()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#import xgboost as xgb
#import lightgbm as lgb
#Using Lasso Regression
lasso = Lasso(alpha=.05)
fit = lasso.fit(X_train, y_train)

#evaluating error
print(mean_squared_error(fit.predict(X_test), y_test))
print(np.round(r2_score(y_test, lasso.predict(X_test))*100,2),'%')
# Using Ridge Regression
ridge = Ridge(alpha = 7)
fit = ridge.fit(X_train,y_train)

#evaluating error
print(mean_squared_error(fit.predict(X_test), y_test))
print(np.round(r2_score(y_test, ridge.predict(X_test))*100,2),'%')
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(X_train,y_train)

predictions = GBoost.predict(X_test)

print(mean_squared_error(GBoost.predict(X_test), y_test))
print(np.round(r2_score(y_test, GBoost.predict(X_test))*100,2),'%')
#Using Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor as rfr


mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
np.random.seed(11111)
model = rfr(n_estimators=300,max_depth=None)

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(mean_squared_error(model.predict(X_test), y_test))
print(np.round(r2_score(y_test, model.predict(X_test))*100,2),'%')