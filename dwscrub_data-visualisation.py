import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
from sklearn.impute import SimpleImputer
train = pd.read_csv('../input/train.csv')
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])
"""
Normal distribution has skewness = 0, kurtosis = 3
kurtosis > 3 => the distribution generates outliers more often than the normal distribution
kurtosis < 3 => the distribution generates outliers less often than the normal distribution
"""
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
"""
Make heatmap of all correlations in the data
"""
corrmat = train.corr() #make correlation matrix for entire data set
f, ax = plt.subplots(figsize=(12, 9)) #set size of graph
sns.heatmap(corrmat, vmax=.8, square=True) #plot heatmap, vmax = max differentiable probability
"""
Produce heatmap of the top k variables most correlated with SalePrice
"""
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index #returns list of k most correlated vars
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(9,9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
"""
Make scatterplots showing the most correlated data that we have decided to take further.
In cases where values are highly correlated eg. GarageCars and GarageArea the variable with a higher correlation
with SalePrice will be used and the other dropped.
YearBuilt is a time series and idk how to handle that properly at so it's gone
FullBath seems dumb, I don't understand it so I'm not gonna use it at this time
"""
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(train[cols], size = 2.5)
plt.show();
"""
Delete the currently obvious outliers (check GrLivArea and SalePrice, two rightmost points do not fit trend)
"""
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)
"""
Replot graphs to see how this affected the data and to ensure we got the right values
"""
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(train[cols], size = 2.5)
plt.show()
"""
This function generates a histogram for the variable and shows its current distribution as well as how a normal dist. would fit the data
The function then generates a qq plot for the data
"""
def normalComp(var):
    sns.distplot(train[var], fit=norm) #shows current distribution of var and attempts to fit a normal distribution to it
    fig = plt.figure()
    res = stats.probplot(train[var], plot=plt) #plot qq plot
def printSkewAndKurt(var):
    print("Skewness: %f" % train[var].skew())
    print("Kurtosis: %f" % train[var].kurt())
train = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']]
normalComp('SalePrice')
"""
A log transformation seems to sigificantly improve the normality of the data, we will keep this
"""
train['SalePrice'] = np.log(train['SalePrice'])
normalComp('SalePrice')
printSkewAndKurt('SalePrice')
"""
This data is not continuous.
Left side of the data needs to be transformed
The log transformation resulted in worse values for both skew and kurtosis
A BC transformation may be useful here
"""
normalComp('OverallQual')
printSkewAndKurt('OverallQual')
"""
Data is not terrible but the qq plot suggests that there is definitely room for improvement on both ends of the distribution
"""
normalComp('GrLivArea')
printSkewAndKurt('GrLivArea')
train['GrLivArea'].min()
"""
There is still a problem with the left tail of the distribution but the right tail has improved greatly.
"""
train['GrLivArea'] = np.log(train['GrLivArea'])
normalComp('GrLivArea')
printSkewAndKurt('GrLivArea')
"""
Hesitant to apply transformations to categorical data, especially when a case can be made for normality.
Suggestions are welcome, perhaps BC is better here (no idea if it is btw)
"""
normalComp('GarageCars')
printSkewAndKurt('GarageCars')
"""
This data could do with a log transformation but has a significant amount of zero values.
Adding one to the data and then taking logs presents a distribution worse than the original data: skewness -5.2, kurtosis 27.8
Investigate with BC
As the 0 value represents no basement, a new binary variable can hold this data and all non zero entries can be transformed
The above approach bears similarity to imputing unknown values and then creating a new value to note the imputation
"""
normalComp('TotalBsmtSF')
printSkewAndKurt('TotalBsmtSF')
train['TotalBsmtSF'].min()
train['noBasement'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['noBasement'] = 1 
train.loc[train['TotalBsmtSF']>0,'noBasement'] = 0
train.loc[train['noBasement']==1,'TotalBsmtSF'] = np.nan
si = SimpleImputer()
copy = train
train = si.fit_transform(train)
train = pd.DataFrame(train)
train.columns = copy.columns

train['TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

normalComp('TotalBsmtSF')
printSkewAndKurt('TotalBsmtSF')
"""
From looking at the final plots we can see that the variables show less heteroscedasticity, they are also more linear
"""
sns.pairplot(train, size = 2.5)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from math import exp, sqrt
from sklearn.metrics import mean_absolute_error
y = train['SalePrice']
y = pd.DataFrame(y)
X = train
del X['SalePrice']
train_X, val_X, train_y, val_y = train_test_split(X,y)
xgb = XGBRegressor(n_estimators=1000, learning_rate = .001,
                      booster = "gbtree", tree_method = "exact", max_depth = 20)

xgb.fit(train_X, train_y, early_stopping_rounds=3,
             eval_set=[(val_X, val_y)], verbose=False)

preds = [exp(pred) for pred in xgb.predict(val_X)]
goals = [exp(goal) for goal in val_y.SalePrice]
print(mean_absolute_error(preds, goals))