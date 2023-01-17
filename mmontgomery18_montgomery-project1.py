import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(20)

train.head()
train.describe()
train.shape
total = train.isnull().sum().sort_values(ascending=False)

total.head(20)
train.count()
import seaborn as sns

sns.distplot(train['SalePrice']);
train['SalePrice'] = np.log(train['SalePrice'])
corr=houses.corr()["SalePrice"]

corr[np.argsort(corr, axis=0)[::-1]]

sns.boxplot(x="YrSold", y="SalePrice", data=train);
plt.figure(figsize = (8, 4))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = train)

xt = plt.xticks(rotation=45)
import matplotlib.pyplot as plt

cm=train[["SalePrice","OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

plt.subplots(figsize=(6, 4))

sns.heatmap(cm, vmax=1, square=False);
cor_dict = cm['SalePrice'].to_dict()

for i in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*i))
cm = train.corr()["SalePrice"].sort_values(ascending=False)

cm.head(20)
cm = train.corr()["SalePrice"]

cm[np.argsort(cm, axis=0)[::-1]]
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
sns.regplot(x = 'OverallQual', y = 'SalePrice', data = train, color = 'Red')
plt.figure(1)

f, axarr = plt.subplots(3, 2, figsize=(14, 12))

price = train.SalePrice.values

axarr[0, 0].scatter(train.GrLivArea.values, price)

axarr[0, 0].set_title('GrLiveArea')

axarr[0, 1].scatter(train.GarageCars.values, price)

axarr[0, 1].set_title('GarageCars')

axarr[1, 0].scatter(train.TotalBsmtSF.values, price)

axarr[1, 0].set_title('TotalBsmtSF')

axarr[1, 1].scatter(train['FullBath'].values, price)

axarr[1, 1].set_title('FullBath')

axarr[2, 0].scatter(train.TotRmsAbvGrd.values, price)

axarr[2, 0].set_title('YearBuilt')

axarr[2, 1].scatter(train.OverallQual.values, price)

axarr[2, 1].set_title('OverallQual')

plt.tight_layout()

plt.show()
plt.figure(1)

f, axarr = plt.subplots(1, 2, figsize=(12, 4))



price = train.SalePrice.values

axarr[0].scatter(train.TotalBsmtSF.values, price)

axarr[0].set_title('TotalBsmtSF')

axarr[1].scatter(train.GrLivArea.values, price)

axarr[1].set_title('GrLiveArea')

plt.tight_layout()

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
sns.boxplot(x = 'GarageCars', y = 'SalePrice',  data = train)

sns.boxplot(x = 'OverallQual', y = 'SalePrice',  data = train)
sns.boxplot(x = 'GarageCars', y = 'TotalBsmtSF',  data = train)
sns.boxplot(x = 'FullBath', y = 'SalePrice',  data = train)



sns.boxplot(x = 'GarageCars', y = 'YearBuilt',  data = train)

train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice']);
from scipy.stats import skew 

skewness = train.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]

skewness.index

skew_features = train[skewness.index]

skew_features.columns

skew_features = np.log1p(skew_features)
all_data = (train.loc[:,'MSSubClass':'SaleCondition'])

all_data = pd.get_dummies(all_data)

all_data.head(20)
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train.head()
train = train.fillna(train.mean())

test = test.fillna(test.mean())
train.dtypes.sample(10)
sns.boxplot(x = 'FullBath', y = 'SalePrice',  data = train)
X = train.drop(['SalePrice'], axis = 1)



# Create array of target variable

y = train['SalePrice']



# Split training data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20,random_state = 101)



# Fit

# Import model

from sklearn.linear_model import Lasso



# Instantiate object

lasso = Lasso()



# Fit model to training data

lasso = lasso.fit(X_train, y_train)



# Predict

y_pred_lasso = lasso.predict(X_test)



# Score It

from sklearn import metrics

print('Linear Regression Performance')

print('MAE',metrics.mean_absolute_error(y_test, y_pred_lasso))

print('MSE',metrics.mean_squared_error(y_test, y_pred_lasso))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_lasso))



X = train.drop(['SalePrice'],axis = 1)



# Create array of target variable y

y =train['SalePrice']



# Feature Selector

# Import

from sklearn.feature_selection import SelectPercentile, f_regression



# Instantiate object

selector_f = SelectPercentile(f_regression, percentile=20)



# Fit and transform

x_best = selector_f.fit_transform(X, y)
support = np.asarray(selector_f.get_support())



features = np.asarray(X.columns.values)

features_with_support = features[support]
best_feat = train[features_with_support]

corr =best_feat.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 

            cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
from scipy import stats

print('Correlation to SalePrice')

print('GrLivArea:',stats.pearsonr(best_feat['GrLivArea'],train['SalePrice'])[0])

print('TotRmsAbvGrd:',stats.pearsonr(best_feat['TotRmsAbvGrd'],train['SalePrice'])[0])

print('--'*40)

print('GarageCars:',stats.pearsonr(best_feat['GarageCars'],train['SalePrice'])[0])

print('GarageArea:',stats.pearsonr(best_feat['GarageArea'],train['SalePrice'])[0])
best_feat = best_feat.drop(['TotRmsAbvGrd','GarageArea'], axis = 1)

best_feat.columns
best_feat = train[features_with_support]

corr =best_feat.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 

            cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
X_best = train[["OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]



# Create array of target variable

y = train['SalePrice']



# Split training data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_best,y, test_size = .20,random_state = 101)



# Fit

from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_best,y)



# Predict

y_pred_rforest = rforest.predict(X_test)



# Score It

from sklearn import metrics

print('Random Forest Regression Performance')

print('MAE',metrics.mean_absolute_error(y_test, y_pred_rforest))

print('MSE',metrics.mean_squared_error(y_test, y_pred_rforest))

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_rforest)))

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_rforest))
z_pred_rforest = rforest.predict(X_train)

print('RMSE',np.sqrt(metrics.mean_squared_error(y_train, z_pred_rforest)))
X_train2 = train[["OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]



# Create target variable array for training data

y_train2 = train["SalePrice"]



# Create matrix of x features for test data

X_test2 = test[["OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]



# There is no target variable array for test data



# Confirm data shapes

print('Data Shapes')

print('x_train shape', X_train2.shape)

print('y_train shape',y_train2.shape)

print('x_test shape', X_test2.shape)



# Fit Random Forest to training data

from sklearn.ensemble import RandomForestRegressor

rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_train2,y_train2)



# Predict using test data

y_pred_rforest2 = rforest.predict(X_test2)

submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": y_pred_rforest2

    })



submission.to_csv('houseprices1.csv', index=False)