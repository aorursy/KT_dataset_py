%matplotlib inline

import numpy as np 

import math as mt

from scipy import stats

from scipy.stats import norm

import pandas as pd # data processing, CSV file I/O 

import matplotlib.pyplot as plt # various graphs

#plt.rcParams['figure.figsize'] = (20.0, 10.0)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge,LinearRegression

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from pandas.plotting import scatter_matrix

import seaborn as sns

import os

import pymc3 as pm

pd.set_option('display.float_format', lambda x: '%.3f' % x)#remove scientific notation
#pd.set_option('display.max_columns',None)

#pd.set_option('display.max_rows',None)

project_dir="../input/"

housing = pd.read_csv(project_dir+"train.csv")

test_set = pd.read_csv(project_dir+"test.csv")
print("Initial Train set: ",housing.shape)

print("Test set: ",test_set.shape)
housing.info()
housing.sample(n=10)
#transfortm to objects (categorical)

housing["OverallQual"]= housing["OverallQual"].astype(object)

housing["OverallCond"]= housing["OverallCond"].astype(object)

housing["MSSubClass"] = housing["MSSubClass"].astype(object)

#transfortm to floats

housing["LotArea"] = housing["LotArea"].astype(float)

housing["BsmtFinSF1"] = housing["BsmtFinSF1"].astype(float) 

housing["BsmtFinSF2"] = housing["BsmtFinSF2"].astype(float) 

housing["BsmtUnfSF"] = housing["BsmtUnfSF"].astype(float) 

housing['TotalBsmtSF'] = housing['TotalBsmtSF'].astype(float)

housing['1stFlrSF'] = housing['1stFlrSF'].astype(float)

housing['2ndFlrSF'] = housing['2ndFlrSF'].astype(float)

housing['LowQualFinSF'] = housing['LowQualFinSF'].astype(float)

housing['GrLivArea'] = housing['GrLivArea'].astype(float)

housing['WoodDeckSF'] = housing['WoodDeckSF'].astype(float)

housing['OpenPorchSF'] = housing['OpenPorchSF'].astype(float)

housing['SalePrice'] = housing['SalePrice'].astype(float)

housing['PoolArea'] = housing['PoolArea'].astype(float)

housing['ScreenPorch'] = housing['ScreenPorch'].astype(float)

housing['GarageArea'] = housing['GarageArea'].astype(float)

housing['3SsnPorch'] = housing['3SsnPorch'].astype(float)
housing.describe(exclude=[np.object])
housing.describe(include=[np.object]) 
housing[housing.duplicated()]
housing.isnull().sum()[housing.isnull().sum()>0]#checking for null data
housing.eq("").sum(axis=0)#checking for spaces
total = housing.isnull().sum().sort_values(ascending=False)

percent = (housing.isnull().sum()/housing.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent Missing'])

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent Missing'])
housing = housing.drop((missing_data[missing_data['Total Missing'] > 1]).index,1)
housing = housing.drop(housing.loc[housing['Electrical'].isnull()].index)

housing.isnull().sum().max() #final check to see if there is no missing data missing... ;)
print("Float Size:",housing.select_dtypes(['float']).columns.size," names:",housing.select_dtypes(['float']).columns)
print("Integer Size:",housing.select_dtypes(['int64']).columns.size," names:",housing.select_dtypes(['int64']).columns)
print("Categorical Size:", housing.select_dtypes(['object']).columns.size," names:", housing.select_dtypes(['object']).columns)
%matplotlib inline

housing.hist(bins=50,figsize=(60,48))

plt.show()
%matplotlib inline

nbins=50

nbins,plt.hist(housing['SalePrice'],nbins,facecolor='blue')

plt.show()
%matplotlib inline

nbins=50

nbins,plt.hist(housing['YearBuilt'],nbins,facecolor='blue')

plt.show()
housing['TotalSF'] =  housing['TotalBsmtSF'] + housing['GrLivArea']

#all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

housing['TotalSF'] = housing['TotalSF'].astype(float)

%matplotlib inline

nbins=50

nbins,plt.hist(housing['TotalSF'],nbins,facecolor='blue')

plt.show()
housing['SaleCondition'].value_counts()
%matplotlib inline

housing['SaleCondition'].value_counts().plot(kind='bar')

plt.yscale('log')

plt.show()
%matplotlib inline

housing['MoSold'].value_counts().plot(kind='bar')

plt.yscale('log')

plt.show()
housing['YrSold'].value_counts().plot(kind='bar')

housing["YrSold"].value_counts()
corr_mat=housing.select_dtypes(['float']).corr()

mask = np.zeros_like(corr_mat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True,center="light")

sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=1,vmin=-1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr_mat["SalePrice"].sort_values(ascending=False)#check the price correlation with each other col
cols =corr_mat["SalePrice"].sort_values(ascending=False).index.tolist()[0:10]#select the first 10 in desceding order variables that have positive correlation

sns.pairplot(housing[cols], height = 1.8)

plt.show()
housing.plot('TotalSF', 'SalePrice', kind = 'scatter', marker = 'o',alpha=.1);
housing.plot('GrLivArea', 'SalePrice', kind = 'scatter', marker = 'o',alpha=.1);
#deleting points based on the GrLivArea (possible outliers check also the TotalSF)

housing.sort_values(by = 'GrLivArea', ascending = False)[:2]#1298 and 523

housing = housing.drop(housing[housing['Id'] == 1299].index)

housing = housing.drop(housing[housing['Id'] == 524].index)
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([housing['SalePrice'], housing[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

ax.set_yscale('log')

fig.axis(ymin=25000, ymax=1000000);
#box plot overallcond/saleprice

var = 'OverallCond'

data = pd.concat([housing['SalePrice'], housing[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

ax.set_yscale('log')

fig.axis(ymin=25000, ymax=1000000);
var = 'YearBuilt'

data = pd.concat([housing['SalePrice'], housing[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=25000, ymax=1000000);

ax.set_yscale('log')

plt.xticks(rotation=90);
#histogram and normal probability plot

sns.distplot(housing['SalePrice'],fit=norm)

fig = plt.figure()

res = stats.probplot(housing['SalePrice'], plot=plt)
housing['SalePrice'] = np.log(housing['SalePrice'])
#histogram and normal probability plot

sns.distplot(housing['SalePrice'],fit=norm)

fig = plt.figure()

res = stats.probplot(housing['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(housing['TotalSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(housing['TotalSF'], plot=plt)
housing['TotalSF'] = np.log(housing['TotalSF'])
#histogram and normal probability plot

sns.distplot(housing['TotalSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(housing['TotalSF'], plot=plt)
plt.scatter(housing['TotalSF'], housing['SalePrice']);
plt.scatter(housing[housing['TotalBsmtSF']>0]['TotalBsmtSF'], housing[housing['TotalBsmtSF']>0]['SalePrice']);
#re-arrange columns in order to be SalePrice in the end

tempo_cols=list(housing.columns)

tempo_cols[-1]="SalePrice"

tempo_cols[-2]="TotalSF"

housing = housing[tempo_cols]

#housing.head(3)
scaler   = StandardScaler()

num_cols = list(housing.select_dtypes(exclude = ["object"]).columns)

cat_cols = list(housing.select_dtypes(include = ["object"]).columns)

del num_cols[-1] #dropping SalePrice

num_cols

housing_scaled = scaler.fit_transform(housing[num_cols])

housing_scaled = pd.DataFrame(housing_scaled, columns=num_cols)

housing_scaled = housing_scaled.join(housing['SalePrice'])#adding non scaled SalePrice

housing_scaled = housing_scaled.drop('Id', 1)#drop column Id # redundant

#housing_scaled = housing[cat_cols].join(housing_scaled)#adding the categorical data (of course not scaled either!)
categorical_features = housing_scaled.select_dtypes(include = ["object"]).columns

numerical_features = housing_scaled.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features   : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))
housing_scaled.head(3)
housing_scaled=pd.get_dummies(housing_scaled)

housing_scaled=housing_scaled.dropna()
categorical_features = housing_scaled.select_dtypes(include = ["object"]).columns

numerical_features = housing_scaled.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

del housing_scaled['BsmtFullBath']

del housing_scaled['BsmtHalfBath']

print("Numerical features   : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))
def split_train_val(data,ratio,seedme):

    np.random.seed(seedme)#in order to have consistency and reproducibility in the results 

    shuffleme = np.random.permutation(len(data))

    val_set_size  = int(len(data)*ratio)

    val_indices   = shuffleme[:val_set_size]

    train_indices = shuffleme[val_set_size:]

    return data.iloc[train_indices],data.iloc[val_indices]

train_set,val_set = split_train_val(housing_scaled,0.2,666)

print("Train set: ",len(train_set),", validation set: ",len(val_set))
x_train = train_set.loc[:, train_set.columns != 'SalePrice']

y_train = train_set.SalePrice

x_val   = val_set.loc[:, val_set.columns != 'SalePrice']

y_val   = val_set.SalePrice

print("x_train : " + str(x_train.shape))

print("x_val : " + str(x_val.shape))

print("y_train : " + str(y_train.shape))

print("y_val : " + str(y_val.shape))
lr = LinearRegression()

lr.fit(x_train, y_train)



y_train_pred = lr.predict(x_train)

y_val_pred   = lr.predict(x_val)

# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_val_pred, y_val_pred - y_val, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_val_pred, y_val, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()

lr.score(x_train,y_train)
rg = Ridge()

rg.fit(x_train,y_train)



Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,

      normalize=False, random_state=None, solver='auto', tol=0.001)



rg.score(x_train,y_train)
sr = SVR(gamma = 'scale',epsilon=0.2)

sr.fit(x_train,y_train)



SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale',

    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



sr.score(x_val,y_val)
rf = RandomForestRegressor(n_jobs=-1) # n_jobs = -1 , to distribute the computaion on diff cpu cores

rf.fit(x_train,y_train)

rf.score(x_val,y_val)
sub_attributes=["SalePrice","GarageArea","TotalSF"]

scatter_matrix(housing[sub_attributes],figsize=(20,15))

plt.show()
house_model = pm.Model()

y = train_set.SalePrice.values

X = train_set.loc[:, val_set.columns != 'SalePrice']

X1 = X['TotalSF']

X2 = X['GarageArea']
with house_model:



    # Priors for unknown model parameters

    

    # Prior for intercept is still mean of observed - different from prior used

    # in our version

    intercept = pm.Normal('intercept', mu=np.mean(y), sd=1000)

    

    # note shape = 2 parameter

    beta = pm.Normal('beta', mu=0, sd=10, shape=2)

    sigma = pm.Normal('sigma',mu = 1000, sd=50)



    # Expected value of outcome

    mu = intercept + beta[0]*X1 + beta[1]*X2



    # Likelihood (sampling distribution) of observations

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

    

    # instantiate sampler

    # Slice is a simpler sampler

    step = pm.Slice()



    # draw 500, the 5000 posterior samples

    trace_500_housing = pm.sample(500, step = step)

    trace_5k_housing = pm.sample(5000, step = step)

    ppc = pm.sample_posterior_predictive(trace_5k_housing)

pm.traceplot(trace_500_housing)
pm.traceplot(trace_5k_housing)
plt.figure()

plt.plot(X1, ppc['Y_obs'].T, '.b', alpha=0.01)

plt.plot(X1, y, 'or')

proxy_arts = [plt.Line2D([0], [0], marker='.', linestyle='', color='b'),

              plt.Line2D([0], [0], marker='o', linestyle='', color='r')]

plt.legend(handles=proxy_arts, labels=['prediction', 'training data'])

plt.title('Posterior predictive on the training set')
list(trace_500_housing.points())[:2]
list(trace_5k_housing.points())[:2]
test_set = pd.read_csv(project_dir+"test.csv")

#transform to objects (categorical)

test_set["OverallQual"]= test_set["OverallQual"].astype(object)

test_set["OverallCond"]= test_set["OverallCond"].astype(object)

test_set["MSSubClass"] = test_set["MSSubClass"].astype(object)

#transfortm to floats

test_set["LotArea"] = test_set["LotArea"].astype(float)

test_set["BsmtFinSF1"] = test_set["BsmtFinSF1"].astype(float) 

test_set["BsmtFinSF2"] = test_set["BsmtFinSF2"].astype(float) 

test_set["BsmtUnfSF"] = test_set["BsmtUnfSF"].astype(float) 

test_set['TotalBsmtSF'] = test_set['TotalBsmtSF'].astype(float)

test_set['1stFlrSF'] = test_set['1stFlrSF'].astype(float)

test_set['2ndFlrSF'] = test_set['2ndFlrSF'].astype(float)

test_set['LowQualFinSF'] = test_set['LowQualFinSF'].astype(float)

test_set['GrLivArea'] = test_set['GrLivArea'].astype(float)

test_set['WoodDeckSF'] = test_set['WoodDeckSF'].astype(float)

test_set['OpenPorchSF'] = test_set['OpenPorchSF'].astype(float)

test_set['PoolArea'] = test_set['PoolArea'].astype(float)

test_set['ScreenPorch'] = test_set['ScreenPorch'].astype(float)

test_set['GarageArea'] = test_set['GarageArea'].astype(float)

test_set['3SsnPorch'] = test_set['3SsnPorch'].astype(float)



total = test_set.isnull().sum().sort_values(ascending=False)

percent = (test_set.isnull().sum()/test_set.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent Missing'])

test_set = test_set.drop((missing_data[missing_data['Total Missing'] > 1]).index,1)

test_set = test_set.drop(test_set.loc[test_set['Electrical'].isnull()].index)

test_set.isnull().sum().max() #final check to see if there is no missing data missing... ;)

print("Float Size:",test_set.select_dtypes(['float']).columns.size," names:",test_set.select_dtypes(['float']).columns)

print("Integer Size:",test_set.select_dtypes(['int64']).columns.size," names:",test_set.select_dtypes(['int64']).columns)

print("Categorical Size:", test_set.select_dtypes(['object']).columns.size," names:", test_set.select_dtypes(['object']).columns)

test_set['TotalSF'] =  test_set['TotalBsmtSF'] + test_set['GrLivArea']

test_set['TotalSF'] = test_set['TotalSF'].astype(float)

test_set['TotalSF'] = np.log(test_set['TotalSF'])



scaler   = StandardScaler()

num_cols = list(test_set.select_dtypes(exclude = ["object"]).columns)

cat_cols = list(test_set.select_dtypes(include = ["object"]).columns)



test_set_scaled = scaler.fit_transform(test_set[num_cols])

test_set_scaled = pd.DataFrame(test_set_scaled, columns=num_cols)



categorical_features = test_set_scaled.select_dtypes(include = ["object"]).columns

numerical_features = test_set_scaled.select_dtypes(exclude = ["object"]).columns

print("Numerical features   : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

test_set_scaled=test_set_scaled.dropna()

test_set_scaled_id=test_set_scaled.Id

del test_set_scaled['Id']



test_set_scaled.head(3)

predicted_prices = rf.predict(test_set_scaled)

predicted_prices = predicted_prices.reshape(-1)

predicted_prices.shape

np.exp(predicted_prices)
my_submission = pd.DataFrame({'Id':test_set_scaled_id, 'SalePrice': np.exp(predicted_prices)})

#my_submission.to_csv('submitme.csv', index=False)
my_submission