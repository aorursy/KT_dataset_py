# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge, Lasso

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_error

from scipy.stats import norm

from scipy import stats

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_columns', None)

pd.options.display.float_format = '{:.2f}'.format 
trdf = pd.read_csv('../input/train.csv')
plt.figure(figsize = (10,10))

sns.heatmap(trdf.isnull(),yticklabels = False, cbar = False, cmap='coolwarm')
#drop missing data

#columns = alley, fireplacequ, poolqc, fence, miscfeatre - Done

trdf = trdf.drop (columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
#Fill missing rows - done

trdf['LotFrontage'].fillna(0, inplace = True)



Glist = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual','GarageYrBlt']

for item in Glist:

    trdf[item].fillna(0, inplace = True)

    



Blist = ['BsmtExposure','BsmtQual','BsmtExposure','BsmtCond','BsmtFinType1','BsmtFinType2']



for item in Blist:

    trdf[item].fillna(0, inplace = True)

    

trdf['MasVnrType'].fillna('None', inplace = True)



trdf['MasVnrArea'].fillna(0, inplace = True)



trdf['Electrical'].fillna('SBrkr',inplace = True)
#Remove Outliers - ID:441

trdf[trdf['TotalBsmtSF']>trdf['GrLivArea']]
#Remove Outliers - ID:1299

trdf[trdf['TotalBsmtSF']>5900]
#Remove Outliers - ID:524 and 1299

trdf[trdf['GrLivArea']>4000]
#confirmed those were it

sns.pairplot(trdf.drop(index = [523,1298,440])[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 

       'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',

       'YearRemodAdd', 'MasVnrArea', 'Fireplaces']])
trdf.drop(index = [523,1298], inplace = True)
#recoding numerical variables as cat

trdf = trdf.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}})
#recoding categorical data as ordinal

trdf = trdf.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
#run a heatmap on all variables to identify collinear variables

plt.figure(figsize = (10,10))

sns.heatmap(trdf.corr())
#drop collinear variables (after checking against corr heatmap)

trdf.drop(columns = ['GarageArea','1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd','YearRemodAdd', 'BsmtCond', 

                   'BsmtFinType1','BsmtFinType2', 'GarageCond' ], inplace = True)
trdf.info()
testdf = pd.read_csv('../input/test.csv')
testdf.head()
testdf.describe()
(testdf.isna().sum()/(testdf.count()+testdf.isna().sum())).sort_values(ascending = False)
#drop missing data

#columns = alley, fireplacequ, poolqc, fence, miscfeatre - Done

df = testdf.drop (columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
#Fill missing rows - done

df['LotFrontage'].fillna(0, inplace = True)



Glist = ['GarageCond', 'GarageType', 'GarageFinish', 'GarageQual','GarageYrBlt']

for item in Glist:

    df[item].fillna(0, inplace = True)

    



Blist = ['BsmtExposure','BsmtQual','BsmtExposure','BsmtCond','BsmtFinType1','BsmtFinType2']



for item in Blist:

    df[item].fillna(0, inplace = True)

    

df['MasVnrType'].fillna('None', inplace = True)



df['MasVnrArea'].fillna(0, inplace = True)



df['Electrical'].fillna('SBrkr',inplace = True)
#recoding numerical variables as cat

df = df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}})
#recoding categorical data as ordinal

df = df.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
#drop collinear variables (after checking against corr heatmap)

df.drop(columns = ['GarageArea','1stFlrSF', 'GarageYrBlt', 'TotRmsAbvGrd','YearRemodAdd', 'BsmtCond', 

                   'BsmtFinType1','BsmtFinType2', 'GarageCond' ], inplace = True)
df.info()
dn = (df.isna().sum()>0)
#input mode for remaining features

for feature in dn[dn].index:

    df[feature].fillna(df[feature].mode()[0], inplace = True)
df.isna().sum()>0
#Need to align and ensure the train and test set are the same size AFTER cleaning, but before log transform



#store test ID for submission purposes

TestId=df['Id']

#align data set shapes and get dummies

total_features=pd.concat((trdf.drop(['Id','SalePrice'], axis=1), df.drop(['Id'], axis=1)))

total_features=pd.get_dummies(total_features, drop_first=True)

train_features=total_features[0:trdf.shape[0]]



#making sure the test set matches the train set

test_features=total_features[trdf.shape[0]:] 
#total features - concat train and test data sets on top of each other without ID and SalePrice to make sure they are the same

total_features.shape
train_features.shape
test_features.shape
#Not normally distributed and needs a transformation. Fails fat pencil test.

sns.distplot(trdf['GrLivArea'],bins = 50, fit=norm)

fig = plt.figure()

stats.probplot(trdf['GrLivArea'], plot=plt)

#skewness and kurtosis

print("Skewness: %f" % trdf['GrLivArea'].skew())

print("Kurtosis: %f" % trdf['GrLivArea'].kurt())
#Not normally distributed and needs a transformation. Fails fat pencil test.

sns.distplot(trdf['SalePrice'],bins = 50, fit=norm)

fig = plt.figure()

stats.probplot(trdf['SalePrice'], plot=plt)

#skewness and kurtosis

print("Skewness: %f" % trdf['SalePrice'].skew())

print("Kurtosis: %f" % trdf['SalePrice'].kurt())
#Perform a Log transformation on GrLivArea

train_features['GrLivArea'] = np.log(train_features['GrLivArea'])

test_features['GrLivArea'] = np.log(test_features['GrLivArea'])
#Perform log transform on SalePrice as it is skewed as well

X = train_features

y = np.log(trdf['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#iterate through possible alpha values to optimize 

rmse=[]

# check the below alpha values for Ridge Regression

alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]



for alph in alpha:

    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)

    ridge.fit(X_train, y_train)

    predict=ridge.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

plt.scatter(alpha, rmse)

rmse = pd.Series(rmse, index = alpha)

print('Best alpha:', rmse.idxmin())
# Adjust alpha based on previous result - can iterate on this to dial it in as much as you need

alpha=np.arange(.5,3, 0.25)



rmse=[]

for alph in alpha:

    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)

    ridge.fit(X_train, y_train)

    predict=ridge.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

plt.scatter(alpha, rmse)

rmse = pd.Series(rmse, index = alpha)

print('Best alpha:', rmse.idxmin())
besta = rmse.idxmin()
#Run your ridge with the highest alpha 

ridge=Ridge(alpha=besta, copy_X=True, fit_intercept=True)

ridge.fit(X_train, y_train)

predictions=ridge.predict(X_test)
# Plot important coefficients 

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()
#plot predictions against actuals

plt.scatter(y_test,predictions)
#plot residuals

sns.distplot((y_test-predictions),bins=500)
#Ridge Explained variance 

explained_variance_score(y_test,predictions)
print('RMSE',np.sqrt(mean_squared_error(y_test, predictions)))
#iterate through possible alpha values to optimize 

rmse=[]

# check the below alpha values for Ridge Regression

alpha=[0.00001, 0.0002, 0.0003, 0.0004, 0.0005, .001, .01, .03, .10, .30, 1]



for alph in alpha:

    lasso=Lasso(alpha=alph, copy_X=True, fit_intercept=True)

    lasso.fit(X_train, y_train)

    predict=lasso.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

plt.scatter(alpha, rmse)

rmse = pd.Series(rmse, index = alpha)

print('Best alpha:', rmse.idxmin())
# Adjust alpha based on previous result - can iterate on this to dial it in as much as you need

alpha=np.arange(.0001,.0004, 0.00005)



rmse=[]

for alph in alpha:

    lasso=Lasso(alpha=alph, copy_X=True, fit_intercept=True, max_iter=10000)

    lasso.fit(X_train, y_train)

    predict=lasso.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

plt.scatter(alpha, rmse)



rmse=pd.Series(rmse, index=alpha)

print('Best alpha:', rmse.idxmin())
besta = pd.Series(rmse, index = alpha).idxmin()
#Run your lasso with the highest alpha 

lasso=Lasso(alpha=besta, copy_X=True, fit_intercept=True)

lasso.fit(X_train, y_train)

predictions=lasso.predict(X_test)
# Plot important coefficients 

coefs = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")

plt.show()
#plot predictions against actuals

plt.scatter(y_test,predictions)
#plot residuals

sns.distplot((y_test-predictions),bins=500)
#Ridge Explained variance - need to beat 90%... beat it with 92%!

explained_variance_score(y_test,predictions)
#Lasso is better on the train dataset than Ridge

print('RMSE',np.sqrt(mean_squared_error(y_test, predictions)))
#Find best alpha and l1 value

from sklearn.linear_model import ElasticNet

rmse=[]

# check the below alpha values for Ridge Regression

alpha = [0.0001, 0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]

l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]

combined = [(alph, l1) for alph in alpha for l1 in l1_ratios]



for c in combined:

    enet=ElasticNet(alpha=c[0], l1_ratio=c[1], normalize=False)

    enet.fit(X_train, y_train)

    predict=enet.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

rmse = pd.DataFrame(rmse, index = combined)

rmse.plot(figsize = (10,10))

plt.xticks(np.arange(len(rmse.index)), rmse.index, rotation = 70)

plt.show()

print('Best alpha, l1:', rmse.idxmin())
#zoom in to better see alpha l1 combo

rmse=[]

# check the below alpha values for Ridge Regression

alpha = [0.0001, 0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]

l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]

combined = [(alph, l1) for alph in alpha for l1 in l1_ratios]



for c in combined[:10]:

    enet=ElasticNet(alpha=c[0], l1_ratio=c[1], normalize=False)

    enet.fit(X_train, y_train)

    predict=enet.predict(X_test)

    rmse.append(np.sqrt(mean_squared_error(y_test, predict)))

print(rmse)

rmse = pd.DataFrame(rmse, index = combined[:10])

rmse.plot(figsize = (10,10))

plt.xticks(np.arange(len(rmse.index)), rmse.index, rotation = 70)

plt.show()

print('Best alpha, l1:', rmse.idxmin())
besta = rmse.idxmin()
#Run your ElasticNet with the highest alpha 

enet=ElasticNet(alpha=besta[0][0], l1_ratio=besta[0][1], normalize=False)

enet.fit(X_train, y_train)

predictions=enet.predict(X_test)
# Plot important coefficients - THIS I LIKE!

coefs = pd.Series(enet.coef_, index = X_train.columns)

print("ENet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the ENet Model")

plt.show()
#plot predictions against actuals

plt.scatter(y_test,predictions)
#plot residuals

sns.distplot((y_test-predictions),bins=500)
#ENet Explained variance 

explained_variance_score(y_test,predictions)
#ENet is better than Lasso and Ridge on the train dataset

print('RMSE',np.sqrt(mean_squared_error(y_test, predictions)))
#create predictions for test set using trained model

#lasso RMSE -  0.12117 

testpredictions=lasso.predict(test_features)

testpredictions
#Ridge RMSE - 0.12144

#testpredictions=ridge.predict(test_features)

#testpredictions
#ENet RMSE - 0.12149

#testpredictions=enet.predict(test_features)

#testpredictions
#transform log sales price back to regular for submission

Test_price=np.exp(list(testpredictions))

Test_price
#create submission

submission=pd.DataFrame()

submission['Id']=TestId

submission['SalePrice']=Test_price

submission.to_csv('submission.csv', index=False)