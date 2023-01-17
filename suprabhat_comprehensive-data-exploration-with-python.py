#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from subprocess import check_output

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#bring in the six packs

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



#check the decoration

df_train.columns
#descriptive statistics summary

df_train['SalePrice'].describe()
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
df_train.describe()
df_train.shape,df_test.shape
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
corrmat = df_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(df_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.barplot(df_train.OverallQual,df_train.SalePrice)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
print("Find most important features relative to target")

corr = df_train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)

#this you can see at the time of heatmap also.
# Differentiate numerical features (minus the target) and categorical features

categorical_features = df_train.select_dtypes(include=['object']).columns

categorical_features
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
numerical_features = df_train.select_dtypes(exclude = ["object"]).columns
# Differentiate numerical features (minus the target) and categorical features

categorical_features = df_train.select_dtypes(include = ["object"]).columns

numerical_features = df_train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_num = df_train[numerical_features]

train_cat = df_train[categorical_features]
# Handle remaining missing values for numerical features by using median as replacement

print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

train_num = train_num.fillna(train_num.median())

print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
from scipy.stats import skew 

skewness = train_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]

skewness.index
skew_features = df_train[skewness.index]

skew_features.columns
#we can treat skewness of a feature with the help fof log transformation.so we'll apply the same here.

skew_features = np.log1p(skew_features)
# Create dummy features for categorical values via one-hot encoding

train_cat.shape 
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot



from scipy import stats

from scipy.stats import norm, skew #for some statistics



sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot

plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)
train_cat.shape

train_cat = pd.get_dummies(train_cat)

train_cat.shape

str(train_cat.isnull().values.sum())
import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.concat([train_cat,train_num],axis=1)

train.shape

y = df_train.SalePrice
#split the data to train the model 

X_train,X_test,y_train,y_test = train_test_split(df_train,y,test_size = 0.3,random_state= 0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train.head(3)

X_train.shape
n_folds = 5

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)

def rmse_CV_test(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)
lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)

print('rmse on train',rmse_CV_train(lr).mean())

print('rmse on train',rmse_CV_test(lr).mean())
#plot between predicted values and residuals

plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre,test_pre - y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
# Plot predictions - Real values

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train,y_train)

alpha = ridge.alpha_

print('best alpha',alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_CV_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)
coef = pd.Series(ridge.coef_, index = X_train.columns)



print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# Plot residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue",  label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "black", marker = "v", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
# Plot predictions - Real values

plt.scatter(y_train_rdg, y_train, c = "blue",  label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()