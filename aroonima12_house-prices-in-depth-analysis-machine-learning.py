# Imports and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from IPython.display import display
pd.options.display.max_columns = None
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.info()

train.describe()
train[(train['LotArea']>11601)&(train['LotArea']<=215245)]['LotArea'].sort_values(ascending =False).head(20)
train['MiscVal'].sort_values(ascending =False).head(20)
# Creating list for numeric values.
from pandas.api.types import is_numeric_dtype
numcollist=[col for col in train.drop('Id', axis=1).columns if is_numeric_dtype(train[col]) ]
for col in numcollist:
    print(col, train[col].nunique())
# Creating list for ordinal and categorical variables

ordlist=[col for col in numcollist if train[col].nunique() < 16 ]
ordlist.append('YearBuilt')
ordlist.append('YearRemodAdd')
ordlist.append('GarageYrBlt')
for col in ordlist:
    numcollist.remove(col)
categlist=[col for col in train.drop('Id', axis=1).columns if not is_numeric_dtype(train[col])]

categlist=categlist+ ordlist


train[categlist].isnull().sum()
# Missing values
train[numcollist].isnull().sum()
#zero values
for col in numcollist:
    print(col,train[train[col]==0][col].count())

    
#Histogram of SalePrice to see impact of zero values in 'BsmtSF1'
train[train['BsmtFinSF1']!=0]['SalePrice'].plot.hist( bins=30, color= 'blue', label='SalePrice at non-zero BsmtSF1')
plt.axvline(train[train['BsmtFinSF1']!=0]['SalePrice'].mean(),color= 'blue', label='Mean Sale Price for Non zero BsmtSF1')
train[train['BsmtFinSF1']==0]['SalePrice'].plot.hist( bins=30, color='yellow',alpha=.6, label='SalePrice at zero BsmtSF1')
plt.axvline(train[train['BsmtFinSF1']==0]['SalePrice'].mean(),color= 'yellow', label='Mean Sale Price for zero BsmtSF1')
plt.xlabel('SalePrice')
plt.title('Saleprice variation at zero values of BsmtFinSF1')
plt.legend()
#Histogram of SalePrice to see impact of zero values in 'BsmtFinSF2'
train[train['BsmtFinSF2']!=0]['SalePrice'].plot.hist( bins=30, color= 'blue',label='SalePrice at non-zero BsmtFinSF2')
plt.axvline(train[train['BsmtFinSF2']!=0]['SalePrice'].mean(),color= 'blue', label='Mean Sale Price for Non zero BsmFintSF2')
train[train['BsmtFinSF2']==0]['SalePrice'].plot.hist( bins=30, color='yellow',alpha=.6, label='SalePrice at zero BsmtFinSF2')
plt.axvline(train[train['BsmtFinSF2']==0]['SalePrice'].mean(),color= 'yellow', label='Mean Sale Price for zero BsmtFinSF2')
plt.xlabel('SalePrice')
plt.title('Saleprice variation at zero values of BsmtFinSF2')
plt.legend()
#Histogram of SalePrice to see impact of zero values in 'OpenPorchSF'
train[train['OpenPorchSF']!=0]['SalePrice'].plot.hist( bins=30, color= 'blue')
plt.axvline(train[train['OpenPorchSF']!=0]['SalePrice'].mean(),color= 'blue', label='Mean Sale Price for Non zero OpenPorchSF')
train[train['OpenPorchSF']==0]['SalePrice'].plot.hist( bins=30, color='yellow',alpha=.6)
plt.axvline(train[train['OpenPorchSF']==0]['SalePrice'].mean(),color= 'yellow', label='Mean Sale Price for zero OpenPorchSF')
plt.xlabel('SalePrice')
plt.title('Saleprice variation at zero values of OpenPorchSF')
plt.legend()
# We derive zscore and plot its histogram.
train['zscore']=(train['SalePrice']-train['SalePrice'].mean())/train['SalePrice'].std()
plt.hist(train['zscore'], bins=30)
plt.axvline(x=3, color = 'red', label='zscore of 3')
plt.xlabel('Zscore')
plt.ylabel('Frequency')
plt.title('Zscore of SalePrice')
plt.legend()



points=len(train[train['zscore'] >=3])
print('{} of SalePrice points are outliers'.format(points))

train=train.drop('zscore', axis=1)
#Calculate correlation coefficients of variables eith SalePrice.
train[numcollist].corr()['SalePrice'].sort_values(ascending=False).head()
# Select GrLivArea and plot scatter plot
plt.scatter(x='GrLivArea', y='SalePrice' ,data=train)
plt.xlabel('GrliveArea')
plt.ylabel('SalePrice')
plt.title('Relation between SalePrice and GrLivArea')
# Sample of normal distribution
normal = np.random.normal(np.mean(train['SalePrice']), np.std(train['SalePrice']), size=len(train))
#Distribution plot of SalePrice alongside kde plot of a normal distribution. 
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(train['SalePrice'], norm_hist=False, kde = True, bins=50)
sns.kdeplot(normal, color='orange', label='normal')
plt.axvline(train['SalePrice'].mean(), color='lightgreen', linewidth=3, label='Mean')
plt.axvline(np.percentile(train['SalePrice'],25), color='brown', linewidth=3, label='Quartiles')
plt.axvline(np.percentile(train['SalePrice'],75), color='brown', linewidth=3)
plt.xlabel('SalePrice',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.title('Histogram of Saleprice',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

#Plotting cumulative distribution of Sale Price and cumulative normal distribution for comparison.
plt.subplot(1,2,2)
plt.hist(normal, cumulative=True, density=True, color='grey',histtype='stepfilled',rwidth=None, bins=50, label='Normal cumulative distribution')
plt.hist(train['SalePrice'], cumulative=True, density=True, color='lightgreen',histtype='stepfilled',rwidth=None,alpha=.5, bins=50, label='Cumulative Frequency of Sale Price')
plt.legend(loc='upper left', fontsize=14)
plt.axvline(np.percentile(normal, 75), color='grey' ,label='75th point in normal distrubution')
plt.axvline(np.percentile(train['SalePrice'], 75), color='green',label='75th point in Sale price distribution')
plt.xlabel('Sale Price',fontsize=14)
plt.ylabel('Cumulative frequency of Sale Price',fontsize=14)
plt.title('Cumulative Distribution Comparison',fontsize=14)
plt.legend(loc='center',fontsize=14)

# We create a sample of lognormal distribution
lognormal = np.random.normal(np.mean(np.log(train['SalePrice'])), np.std(np.log(train['SalePrice'])), size=len(train))
# Distribution plot of SalePrice alongside kde plot of a lognormal distribution. 
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(np.log(train['SalePrice']), norm_hist=False, kde = True, bins=50, label='Sale Price')
sns.kdeplot(lognormal, color='orange', label='lognormal')
plt.axvline(np.log(train['SalePrice']).mean(), color='lightgreen', linewidth=3, label='Mean Sale Price')
plt.axvline(np.percentile(np.log(train['SalePrice']),25), color='brown', linewidth=3, label='Quartiles')
plt.axvline(np.percentile(np.log(train['SalePrice']),75), color='brown', linewidth=3)
plt.xlabel('SalePrice',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.title('Histogram of Saleprice',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

#Plotting cumulative distribution of Sale Price and cumulative lognormal distribution for comparison.
plt.subplot(1,2,2)
#plt.figure(figsize=(5,5))
plt.hist(lognormal, cumulative=True, density=True, color='grey',histtype='stepfilled',rwidth=None, bins=50, label='Cumulative lognormal distribution')
plt.hist(np.log(train['SalePrice']), cumulative=True, density=True, color='lightgreen',histtype='stepfilled',rwidth=None,alpha=.5, bins=50, label='Cumulative Frequency of Sale Price')
plt.legend(loc='center right', fontsize=14)
plt.xlabel('Sale Price',fontsize=14)
plt.ylabel('Cumulative frequency of Sale Price',fontsize=14)
plt.title('Cumulative Distribution Comparison',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
def plot_variable(col):
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    sns.distplot(train[col], norm_hist=False, kde = True, bins=50)
    plt.axvline(train[col].mean(), color='lightgreen', linewidth=3, label='Mean')
    plt.axvline(np.percentile(train[col],25), color='brown', linewidth=3, label='Quartiles')
    plt.axvline(np.percentile(train[col],75), color='brown', linewidth=3)
    plt.xlabel(col,fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.title('Histogram of ' +col,fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
     
    plt.subplot(1,2,2)    
    sns.boxplot(y=train[col])
    
    
plot_variable('TotalBsmtSF')
plot_variable('GrLivArea')
plot_variable('1stFlrSF')
plot_variable('2ndFlrSF')
plot_variable('GarageArea')
plot_variable('BsmtFinSF1')
plot_variable('MasVnrArea')
plot_variable('LotFrontage')
plot_variable('LotArea')
plot_variable('BsmtFinSF1')
plot_variable('BsmtUnfSF')
plot_variable('WoodDeckSF')
plot_variable('OpenPorchSF')
outliers2=train[(train['BsmtFinSF1']>5000)  | (train['LotFrontage']>300) |  (train['TotalBsmtSF']>5000) | (train['1stFlrSF']>5000) ].index.to_list()
# Calculating correlation coefficients between SalePrice and Numerical independent variables
train[numcollist].corr()['SalePrice'].sort_values(ascending=False)
#Standardising dataframe.
scaleddf=(train[numcollist]-train[numcollist].mean())/train[numcollist].std()

# Regression Scatter plots of 'Saleprice with numerical independant variables from Standardised dataframe using PairGrid'
g = sns.PairGrid(data=scaleddf, x_vars=numcollist[0:6],  y_vars='SalePrice')
g.map(sns.regplot)
plt.xlim((-1,10))
plt.ylim((-3,8))

g = sns.PairGrid(data=scaleddf, x_vars=numcollist[6:12],  y_vars='SalePrice')
g.map(sns.regplot)
plt.xlim((-1,10))
plt.ylim((-3,8))
g = sns.PairGrid(data=scaleddf, x_vars=numcollist[12:18],  y_vars='SalePrice')
g.map(sns.regplot)
plt.xlim((-1,10))
plt.ylim((-3,8))
# Plotting heatmap
plt.figure(figsize=(18,18))
sns.heatmap(train[numcollist].corr(),cmap='coolwarm',annot=True)
plt.title('Correlation Matrix of Numeric variables',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Plotting clustermap
plt.figure(figsize=(18,18))
sns.clustermap(train[numcollist].corr(), cmap='rainbow')
plt.title('Correlation Clustermap of Numeric variables',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Plotting regression plot of GrLivArea and SalePrice.
sns.regplot('GrLivArea', 'SalePrice', data=train)
plt.xlabel('GrLivArea',fontsize=14)
plt.ylabel('SalePrice',fontsize=14)
plt.title('Regression plot of GrLivArea and SalePrice',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Creating new variable
train['newsqvar']=train['TotalBsmtSF']+train['1stFlrSF']+train['GrLivArea']
# Regression plot of SalePrice with new variable post log transformation
sns.regplot(np.log(train['newsqvar']), np.log(train['SalePrice']), data=train)
plt.xlabel('newsqvar',fontsize=14)
plt.ylabel('SalePrice',fontsize=14)
plt.title('Regression plot of SalePrice with new variable post log transformation',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#Calculating correlation of newsqvar with SalePrice
train[['SalePrice','newsqvar']].corr()
# The features need to be adjusted as follows
droplist=['TotalBsmtSF','1stFlrSF','GrLivArea']
train=train.drop(droplist, axis=1)
for col in droplist:
    numcollist.remove(col)
numcollist.append('newsqvar')

#Plotting histogram
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.distplot(train['newsqvar'], norm_hist=False, kde = True, bins=50)
plt.xlabel('newsqvar',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Histogram of Newsqvar',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(1,2,2)
sns.distplot(np.log(train['newsqvar']), norm_hist=False, kde = True, bins=50)
plt.ylabel('Frequency')
plt.xlabel('newsqvar',fontsize=12)
plt.ylabel('Frequency',fontsize=12)
plt.title('Histogram of Newsqvar',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

def plot_catvariable(col):
    plt.figure(figsize=(12,8))
    sns.swarmplot( col,  'SalePrice', data=train)
    plt.xlabel(col,fontsize=12)
    plt.ylabel('SalePrice',fontsize=12)
    plt.title('Swarm Plot of SalePrice vs '+col,fontsize=12)
    plt.xticks(rotation=60,fontsize=12)
    plt.yticks(fontsize=12)
    
plot_catvariable('Neighborhood')
plot_catvariable('ExterQual')
plot_catvariable('ExterCond')
plot_catvariable('Foundation')
plot_catvariable('BsmtQual')
plot_catvariable('HeatingQC')
plot_catvariable('CentralAir')
plot_catvariable('BsmtFinType1')
plot_catvariable('KitchenQual')
plot_catvariable('FullBath')
plot_catvariable('BedroomAbvGr')
plot_catvariable('TotRmsAbvGrd')
# Creating new variables out of numeric variables that reflect absence or presence of zero.
for col in numcollist:
    if 0 in train[col].values:
        #print(col)
        train['zero'+col]=np.where(train[col]==0,1,0)
        categlist.append('zero'+col)
# Nan Imputation
train[numcollist]=train[numcollist].replace(0, np.nan).interpolate(kind='linear',limit_direction='both')
train[categlist]=train[categlist].fillna('Absent')
# Function for assigning numeric labels for categorical variables
def train_category_conversion(col):
    a=train.groupby(col)['SalePrice'].mean()
    a=round((a/10000),2)
    index=a.index.values.tolist()
    weight=a.values.tolist()
    zipped=list(zip(index,weight))
    dict1=dict(zipped)
    train[col]=train[col].map(dict1)

# Function for labelling categorical variables with numeric values for test data.
def test_category_conversion(col):
    a=test.groupby(col)['SalePrice'].mean()
    a=round((a/10000),2)
    index=a.index.values.tolist()
    weight=a.values.tolist()
    zipped=list(zip(index,weight))
    dict1=dict(zipped)
    test[col]=test[col].map(dict1)

# Passing function to provide numeric labels to categorical variables
for col in categlist:
    train_category_conversion(col)
# We create Lasso Regression  object with tuned alpha and fit the data. The data is converted to its logarithmic equivalent.
from sklearn.linear_model import Lasso
X=np.log(train.drop(['SalePrice', 'Id'], axis=1))
y=np.log(train['SalePrice'])
lasso=Lasso(alpha=.001 , random_state=40)
lasso.fit(X,y)


# Now we create a dataframe of coefficients and view the top 20 of them
lasso_features=pd.DataFrame({'features': X.columns, 'coefficients': lasso.coef_}).sort_values(by='coefficients', ascending=False)
lasso_features.head(37)


# We create a list of features whise coefficients are non zero.
featurecols=lasso_features[lasso_features['coefficients']!=0]['features'].values.tolist()
len(featurecols)
plt.figure(figsize=(12,12))
ax=sns.barplot(x='coefficients',y='features', data=lasso_features.head(20) )
ax.set_xlabel('coefficients', fontsize=14)
ax.set_ylabel('features', fontsize=14)
ax.tick_params(axis="y", labelsize=18)
ax.set_title('Lasso Features- Highest coefficients', fontsize=14)

import statsmodels.api as sm
from statsmodels.formula.api import ols
for col  in featurecols:
    print(col,' + ', end='')
#Creating a dataframe olslogtrain
logtrain=np.log(train)
olslogtrain=logtrain
olslogtrain.rename(columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSf","3SsnPorch": "ThreeSsnPorch" }, inplace=True)
# Fitting ols model to data
olsmodel1=ols('SalePrice ~  MSZoning  + LotArea  + Neighborhood  + Condition1  + OverallQual    + YearRemodAdd       + BsmtExposure    + BsmtUnfSF  + HeatingQC  + CentralAir    + FullBath  + HalfBath  + BedroomAbvGr  + KitchenQual  + TotRmsAbvGrd  + Functional      + GarageCars    + WoodDeckSF    + SaleCondition  + newsqvar ',olslogtrain).fit()
print(olsmodel1.summary())
# The list of features are reduced on the basis of above findings
dropcols=['MSSubClass','MasVnrArea' , 'LotFrontage','YearBuilt','Exterior1st','BsmtQual','BsmtFinSF1','2ndFlrSF' ,'Fireplaces','FireplaceQu','GarageYrBlt','GarageFinish','GarageQual','GarageCond' ,'EnclosedPorch' ,'ScreenPorch' ,'MiscVal']  
for col in dropcols:
    featurecols.remove(col)

# Adjusting Lasso_features dataframe to view only statistically significant features.
lasso_features=lasso_features[lasso_features['coefficients']!=0]
lasso_features=lasso_features.set_index('features')
lasso_features=lasso_features.drop(dropcols, axis=0)
lasso_features=lasso_features.reset_index()
# Let's plot the top 20 statistically significant features as per their coefficients.
plt.figure(figsize=(12,12))
ax=sns.barplot(x='coefficients',y='features', data=lasso_features.head(20) )
ax.set_xlabel('Coefficients', fontsize=14)
ax.set_ylabel('Features', fontsize=14)
ax.tick_params(axis="y", labelsize=18)
ax.set_title('Features - statistically significant', fontsize=14)

featurecols
# Deleting outliers.  

train_lm=train.drop(outliers2, axis=0)

# Creating a log object and transforming Features and target variable
from sklearn.preprocessing import FunctionTransformer
logfunc=FunctionTransformer(np.log, inverse_func=np.exp)
logX_lm=logfunc.fit(train_lm[featurecols].values)
logy_lm=logfunc.fit(train_lm['SalePrice'].values)
X_train_lm=logX_lm.transform(train_lm[featurecols].values)
y_train_lm=logy_lm.transform(train_lm['SalePrice'].values)
# with dropped outliers
logX_netrf=logfunc.fit(train.drop(outliers2).drop(['SalePrice','Id'], axis=1).values)
logy_netrf=logfunc.fit(train.drop(outliers2)['SalePrice'].values)
X_train_netrf=logX_netrf.transform(train.drop(outliers2).drop(['SalePrice','Id'], axis=1).values)
y_train_netrf=logy_netrf.transform(train.drop(outliers2)['SalePrice'].values)

#Reading Test data and merging it with sample_submission 
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df2=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
test=pd.merge(test,df2, on='Id')

# Creating New variable'newsqvar' and amending the dataframe
test['newsqvar']=test['TotalBsmtSF']+test['1stFlrSF']+test['GrLivArea']
droplist=['TotalBsmtSF','1stFlrSF','GrLivArea']
test=test.drop(droplist, axis=1)
# Creating new variables out of numeric variables that reflect absence or presence of zero.
for col in numcollist:
    if 0 in test[col].values:
        test['zero'+col]=np.where(test[col]==0,1,0)

# Nan Imputation 
test[numcollist]=test[numcollist].replace(0, np.nan).interpolate(kind='linear',limit_direction='both')
test[categlist]=test[categlist].fillna('Absent')

# Perform numeric labelling of categorical variables for test data

for col in categlist:
    test_category_conversion(col)
# Create features and target set from test data
X_test_lm=logX_lm.transform(test[featurecols].values)
y_test_lm=logy_lm.transform(test['SalePrice'].values)
# with dropped outliers

X_test_netrf=logX_netrf.transform(test.drop(['SalePrice','Id'], axis=1).values)
y_test_netrf=logy_netrf.transform(test['SalePrice'].values)
# Score function parameter
from sklearn.metrics import mean_squared_error, make_scorer
mean_sq_err=make_scorer(mean_squared_error)
# Create and fit Linear Regression object and compute cross validated score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lm1=LinearRegression()
lm1.fit(X_train_lm,y_train_lm)
cv_results=cross_val_score(lm1, X_train_lm, y_train_lm, scoring=mean_sq_err, cv=5)
print('The cross validated RMSE of Linear Regression Model- lm1, on train set is: ',round(np.sqrt(np.mean(cv_results)),3))
prediction=lm1.predict(X_train_lm)
print('The RMSE i.e root mean squared error on Train set is : ', round(np.sqrt(mean_squared_error(y_train_lm, prediction)),3))
prediction=lm1.predict(X_test_lm)
print('The RMSE i.e root mean squared error on Test Set is: ', round(np.sqrt(mean_squared_error(y_test_lm, prediction)),3))
# We perform Grid serach cv for hyperparameter tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

param_grid={'alpha':[0,.1,.3,.5,1,10,50]}
ridg_cv=GridSearchCV(Ridge(),param_grid,scoring=mean_sq_err, cv=5 )
ridg_cv.fit(X_train_lm, y_train_lm)

print(' Optimum value of alpha for Ridge Regression Model is: ' , ridg_cv.best_params_)
print(' The cross validated RMSE for Ridge Regression Model, on train set, using optimum value for alpha is :', np.sqrt(ridg_cv.best_score_).round(3))
# We create Ridge Regression object using optimum value for alpha and fit on train set.

ridg=Ridge(alpha =50)
ridg.fit(X_train_lm, y_train_lm)

prediction=ridg.predict(X_train_lm)
print('The RMSE for Ridge Regression Model on train set is :  ', round(np.sqrt(mean_squared_error(y_train_lm, prediction)),3))

prediction=ridg.predict(X_test_lm)
print('The RMSE for Ridge Regression Model on test set is :  ', round(np.sqrt(mean_squared_error(y_test_lm, prediction)),3))

# Let's apply SVM to the dataset. Here we use default parameters.
from sklearn.svm import SVR
svm_reg=SVR()
svm_reg.get_params
svm_reg.fit(X_train_netrf, y_train_netrf)
prediction= svm_reg.predict(X_train_netrf)

cv_results=cross_val_score(svm_reg, X_train_netrf, y_train_netrf, scoring=mean_sq_err, cv=5)
print('The cross validated for SVR model on Train set is: ',round(np.sqrt(np.mean(cv_results)),3))
print('The RMSE for SVR model on train set is :  ', round(np.sqrt(mean_squared_error(y_train_netrf, prediction)),3))
prediction= svm_reg.predict(X_test_netrf)
print('The RMSE for SVR model on test set is :  ', round(np.sqrt(mean_squared_error(y_test_netrf, prediction)),3))
# Finding the optimum hyperparameters
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid={'max_features':[5,10,15,25], 'max_samples':[200, 300, 500, 700, 1000, 1168],'n_estimators':[100, 500, 1000,1500],'min_samples_split':[2,6,8,10,12,15]}
mean_sq_err=make_scorer(mean_squared_error)

rf_cv=RandomizedSearchCV(RandomForestRegressor(),param_grid,scoring=mean_sq_err, cv=5, random_state=42 )
rf_cv.fit(X_train_netrf,y_train_netrf)

print(' Optimum values of hyperparameters for RandomForest model are: ' , rf_cv.best_params_)
print(' The cross validated score on train set for RandomForest model is : ' , np.sqrt(rf_cv.best_score_).round(3))

from sklearn.ensemble import RandomForestRegressor
# Create and fit Random Forest with optimum hyperparameters
rf =RandomForestRegressor(n_estimators=500,min_samples_split= 6,max_samples=300,max_features=5, random_state=42)
rf.fit(X_train_netrf, y_train_netrf)

prediction=rf.predict(X_train_netrf)
print('The RMSE for Random Forest Model on train set is :  ', round(np.sqrt(mean_squared_error(y_train_netrf, prediction)),3))

prediction=rf.predict(X_test_netrf)
print('The RMSE for Random Forest Model on test set is :  ', round(np.sqrt(mean_squared_error(y_test_netrf, prediction)),3))

final_prediction=prediction
# Create dataframe of Feature importances and list the top 20 features
feature_list=train.drop(['SalePrice', 'Id'], axis=1).columns
df=pd.DataFrame({'Feature_name': feature_list, 'Feature_Importance':rf.feature_importances_.round(2) })
df=df.sort_values(by='Feature_Importance', ascending=False).head(20)
plt.figure(figsize=(12,12))
ax=sns.barplot(x='Feature_Importance',y='Feature_name',data=df.head(20))
ax.set_xlabel('Feature_Importance', fontsize=14)
ax.set_ylabel('Feature_name', fontsize=14)
ax.tick_params(axis="y", labelsize=18)
ax.set_title('Important Features- Random Forest Model', fontsize=14)

# Finding the optimum hyperparameters

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor


param_grid={'min_samples_leaf':[2,5,10,15],'n_estimators':[100,200, 500],'min_samples_split':[2,6,8,10,12,15], 'learning_rate':np.linspace(.05,.15,5), 'max_depth': [2,4,6,8,10,12,15], 'max_features':[5,10,15] }    
mean_sq_err=make_scorer(mean_squared_error)

gb_cv=RandomizedSearchCV(GradientBoostingRegressor(),param_grid,scoring=mean_sq_err, cv=5, random_state=3 )
gb_cv.fit(X_train_netrf,y_train_netrf)

print(' Optimum values of hyperparameters for Gradient Boosting model are: ' , gb_cv.best_params_)
print('The cross validated score for Gradient Boosting model on train set is', np.sqrt(gb_cv.best_score_).round(3))
from sklearn.ensemble import GradientBoostingRegressor
# Create and fit Gradient Boosting Regressor with optimum hyperparameters
gb=GradientBoostingRegressor(n_estimators= 100, min_samples_split= 10, min_samples_leaf=10, max_depth= 2, learning_rate= 0.1, max_features=5, random_state=3)
gb.fit(X_train_netrf, y_train_netrf)
prediction=gb.predict(X_train_netrf)
print('The RMSE for Gradient Boosting model on train set is :  ', round(np.sqrt(mean_squared_error(y_train_netrf, prediction)),3))
prediction=gb.predict(X_test_netrf)
print('The RMSE for Gradient Boosting model on test set is :  ', round(np.sqrt(mean_squared_error(y_test_netrf, prediction)),3))

submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
saleprice=logy_netrf.inverse_transform(final_prediction)
np.sqrt(mean_squared_error(np.log(submission['SalePrice'].values[1000:1400]), np.log(saleprice)[1000:1400]))
np.sqrt(mean_squared_error(submission['SalePrice'].values, saleprice))
submission['SalePrice']=saleprice
submission.to_csv('sample_submission1.csv', index=False)
