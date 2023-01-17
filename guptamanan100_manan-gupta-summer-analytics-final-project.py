# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train.head(5)
test.head(5)
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn
print(train.iloc[:,-1].describe())
seaborn.distplot(train['SalePrice'])
train.iloc[:,-1].isnull().sum()

#qqplot
from scipy import stats
import pylab
stats.probplot(train.iloc[:,-1],dist="norm",plot=pylab)
pylab.show
train.describe(include='all')
#Finding all columns with null values
train.columns[train.isnull().any()].tolist()
#Function for Finding and replacing mmissing values with median
def num_miss(df_in, col_name):
    m = df_in[col_name].describe()['50%']
    df_in.loc[(df_in[col_name].isnull()),col_name] = m
    return df_in

num_miss(train, 'LotFrontage')
num_miss(train, 'MasVnrArea')
num_miss(train, 'GarageYrBlt')
train.columns[train.isnull().any()].tolist()
#Function for Finding and replacing mmissing values with most appearing value
def str_miss(df_in, col_name):
    m = df_in[col_name].describe()['top']
    df_in.loc[(df_in[col_name].isnull()),col_name] = m
    return df_in

str_miss(train,'Alley')
str_miss(train,'MasVnrType')
str_miss(train,'BsmtQual')
str_miss(train,'BsmtCond')
str_miss(train,'BsmtExposure')
str_miss(train,'BsmtFinType1')
str_miss(train,'BsmtFinType2')
str_miss(train,'Electrical')
str_miss(train,'FireplaceQu')
str_miss(train,'GarageType')
str_miss(train,'GarageFinish')
str_miss(train,'GarageQual')
str_miss(train,'GarageCond')
str_miss(train,'PoolQC')
str_miss(train,'Fence')
str_miss(train,'MiscFeature')

train.columns[train.isnull().any()].tolist()
train.describe()
#Function for Finding and replacing outliers with mean
def change_outlier(df_in, col_name):
    q1 = df_in[col_name].describe()['25%']
    q3 = df_in[col_name].describe()['75%']
    m = df_in[col_name].describe()['mean']
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-(1.5*iqr)
    fence_high = q3+(1.5*iqr)
    df_in.loc[(df_in[col_name] <= fence_low) | (df_in[col_name] >= fence_high),col_name] = m
    return df_in
change_outlier(train,'GarageArea')
change_outlier(train,'MasVnrArea')
train.head()
#log transform the target 
train["SalePriceLog"] = np.log1p(train["SalePrice"])
print(train.iloc[:,-1].describe())
seaborn.distplot(train['SalePrice'])
#qqplot
stats.probplot(train.iloc[:,-1],dist="norm",plot=pylab)
pylab.show
#something else that i found online. It reports the skewness.
print("skewness of price: %f" % train['SalePrice'].skew())
print("skewness of log of price: %f" % train['SalePriceLog'].skew())
#this gives a good idea on which numerical parameters does the salesprice depend on
train.corr().iloc[-1,:]
numer = train._get_numeric_data()
numername = numer.columns.values.tolist()
for name in numername:
    plt.scatter(train.SalePrice,train[name])
    plt.legend()
    plt.show()
allname = train.select_dtypes(include='object').columns.values.tolist()
for name in allname:
    train.boxplot(column = 'SalePrice', by = name)
    plt.show()
targetLog = train.iloc[:,-1]
target = train.iloc[:,-2]
del train['SalePrice']
del train['SalePriceLog']
del train['Id']
train.shape
#now  train has all the predictors and target and targetLog have the fianl values
#splitting train into train and test to see how well model works
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2)

#First Baseline Model with SalePrice as Predictor
predicted = y_train.mean()
size = y_test.size
sum =0;
for i in range(size):
    sum = sum + ((y_test.iloc[i] - predicted)*(y_test.iloc[i] - predicted))
mse = sum/size
rmse = mse**0.5
rmse
#splitting train into train and test to see how well model works
X_train, X_test, y_train, y_test = train_test_split(train, targetLog, test_size=0.2)

#Second Baseline Model with SalePriceLog as Predictor
mean = y_train.mean()
predicted = np.expm1(mean)
size = y_test.size
sum =0;
for i in range(size):
    sum = sum + ((y_test.iloc[i] - predicted)*(y_test.iloc[i] - predicted))
mse = sum/size
rmse = mse**0.5
rmse
#Dummy coding for categorical Variables
train = pd.get_dummies(train)
print(train.shape)

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
#Making the first model
from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(X_train,y_train)
print("Mean squared error in Test:",np.mean((model1.predict(X_test) - y_test) ** 2))
print('R² of Test:',model1.score(X_test, y_test))
print("Mean squared error in train:",np.mean((model1.predict(X_train) - y_train) ** 2))
print('R² of train:',model1.score(X_train, y_train))
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(train, targetLog, test_size=0.2)
model2=LinearRegression()
model2.fit(X_train_log,y_train_log)
print("Mean squared error in Test:",np.mean((model2.predict(X_test_log) - y_test_log) ** 2))
print('R² of Test:',model2.score(X_test_log, y_test_log))
print("Mean squared error in train:",np.mean((model2.predict(X_train_log) - y_train_log) ** 2))
print('R² of train:',model2.score(X_train_log, y_train_log))
#Residual vs fitted plot for training set for second model
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = seaborn.residplot(model2.predict(X_train_log), y_train_log, lowess=True,scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted - Train Model 2')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plt.show()

#Residual vs fitted plot for test set for second model
plot_lm_2 = plt.figure(1)
plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0] = seaborn.residplot(model2.predict(X_test_log), y_test_log,lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_2.axes[0].set_title('Residuals vs Fitted - Test Model 2')
plot_lm_2.axes[0].set_xlabel('Fitted values')
plot_lm_2.axes[0].set_ylabel('Residuals')
plt.show()
#Residual vs fitted plot for training set for first model
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = seaborn.residplot(model1.predict(X_train), y_train, lowess=True,scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted - Train Model 1')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plt.show()

#Residual vs fitted plot for test set for first model
plot_lm_2 = plt.figure(1)
plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0] = seaborn.residplot(model1.predict(X_test), y_test,lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_2.axes[0].set_title('Residuals vs Fitted - Test Model 1')
plot_lm_2.axes[0].set_xlabel('Fitted values')
plot_lm_2.axes[0].set_ylabel('Residuals')
plt.show()
from sklearn.model_selection import cross_val_score
lm  = LinearRegression()
score = (np.sqrt(cross_val_score(lm, train, targetLog, cv=10, scoring='neg_mean_squared_error') * -1)).mean()
print (score)
all_cols = train.columns.values.tolist()
included_cols = train.columns.values.tolist()
for i in all_cols:
    prevscore = score
    included_cols.remove(i)
    score = (np.sqrt(cross_val_score(lm, train[included_cols], targetLog, cv=10, scoring='neg_mean_squared_error') * -1)).mean()
    if (score>prevscore):
        included_cols.append(i)
        print('reverted')
final_score = (np.sqrt(cross_val_score(lm, train[included_cols], targetLog, cv=10, scoring='neg_mean_squared_error') * -1)).mean()
final_score
#List of all the columns to be used to train for LinearRegression
print (len(included_cols))
included_cols
#Building the regression Model using the list of columns found using cross validation.
to_use = train[included_cols]
model3 = LinearRegression()
model3.fit(to_use,targetLog)
print("Root Mean squared error:",np.sqrt(np.mean((model3.predict(to_use) - targetLog) ** 2)))
print('R²:',model3.score(to_use, targetLog))
from sklearn.linear_model import Ridge
## training the model
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2,0.5,0.05, 1, 5, 10, 20]
for i in alpha_ridge:
    ridgeReg = Ridge(alpha=i, normalize=True)
    ridgeReg.fit(train,targetLog)
    pred = ridgeReg.predict(train)
    print("Root Mean squared error for ",i," is:",np.sqrt(np.mean((pred - targetLog) ** 2)))
ridgeReg = Ridge(alpha=0.001, normalize=True)
ridgeReg.fit(train,targetLog)
pred = ridgeReg.predict(train)
print("Root Mean squared error is:",np.sqrt(np.mean((pred - targetLog) ** 2)))
from sklearn.linear_model import Lasso
lasso_ridge = [0.0005,1e-3,1e-2,0.5,0.05, 1, 5, 10, 20]
for i in lasso_ridge:
    lassoReg = Lasso(alpha=i, normalize=True)
    lassoReg.fit(train,targetLog)
    pred = lassoReg.predict(train)
    print("Root Mean squared error for ",i," is:",np.sqrt(np.mean((pred - targetLog) ** 2)))
lassoReg = Lasso(alpha=0.0005, normalize=True)
lassoReg.fit(train,targetLog)
pred = lassoReg.predict(train)
print("Root Mean squared error is:",np.sqrt(np.mean((pred - targetLog) ** 2)))
#inputting the files again and concatenating
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
del train['SalePrice']
print(train.shape)
print(test.shape)

frames = [test,train]
conc  = pd.concat(frames)
test.head()
#Finding columns with missing values and replacing using functions defined above
missing = conc.columns[conc.isnull().any()].tolist()
for i in missing:
    types = conc[i].dtype
    if (types == object):
        str_miss(conc,i)
    else:
        num_miss(conc, i)
#Creating dummy variables
conc = pd.get_dummies(conc)
print(conc.shape)
#Getting the test set back
test = conc.iloc[0:1459,:]
test.head()
#Dropping the id column in test
del test['Id']
test.shape
LogPredicted = lassoReg.predict(test)
Predicted = np.expm1(LogPredicted)
Predicted
test = pd.read_csv("../input/test.csv")
test['PredictedSalePrice'] = list(Predicted)
test.head()