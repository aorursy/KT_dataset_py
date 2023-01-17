import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test.head()
print("train shape: ", train.shape)

print("test shape: ", test.shape)
full_data = [train,test]
test_ID = test['Id']



for dataset in full_data:

    dataset.drop('Id',axis=1,inplace=True)
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
train['SalePrice'].describe()
from scipy import stats

from scipy.stats import norm, skew



ax = sns.distplot(train['SalePrice'],fit = norm)

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'],alpha=0.5)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'],alpha=0.5)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])

ax = sns.distplot(train['SalePrice'],fit=norm)

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

train_labels = train.pop('SalePrice')
features = pd.concat([train, test], keys=['train', 'test'])
features.select_dtypes(include='object').isnull().sum()[features.select_dtypes(include='object').isnull().sum()>0]
features.select_dtypes(include=['int','float']).isnull().sum()[features.select_dtypes(include=['int','float']).isnull().sum()>0]
all_data_na = (features.isnull().sum() / len(features)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
features.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'],axis=1, inplace=True)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    features[col] = features[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    features[col] = features[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType'):

    features[col] = features[col].fillna('None')
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional','Electrical','Utilities'):

    features[col]=features[col].fillna(features[col].mode()[0])
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())
print(features.isnull().sum().sum())
from sklearn.preprocessing import LabelEncoder

list_of_col = list(features.select_dtypes(include='object').columns)



for col in list_of_col:

    lbl = LabelEncoder() 

    lbl.fit(list(features[col].values)) 

    features[col] = lbl.transform(list(features[col].values))



# shape        

print('Shape all_data: {}'.format(features.shape))
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]

numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
ax = sns.pairplot(numeric_features_standardized)
all_numeric_feats = features.dtypes[features.dtypes != "object"].index

skewed_feats = features[all_numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    features[feat] = boxcox1p(features[feat], lam)

    

features[skewed_features] = np.log1p(features[skewed_features])
train_features = features.loc['train'].select_dtypes(include=[np.number])

test_features = features.loc['test'].select_dtypes(include=[np.number])
from sklearn.model_selection import KFold, cross_val_score, train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.15, random_state=42)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.pipeline import make_pipeline

from sklearn import linear_model,svm, ensemble

from sklearn.preprocessing import RobustScaler



lasso = make_pipeline(RobustScaler(), linear_model.Lasso(alpha =0.005, random_state=42)).fit(x_train, y_train)

ridge = linear_model.Ridge(alpha = 0.2, random_state=42).fit(x_train, y_train)

bayesian = linear_model.BayesianRidge(n_iter=300).fit(x_train, y_train)

svr = svm.SVR(kernel="linear").fit(x_train, y_train)

gbr = ensemble.GradientBoostingRegressor(n_estimators= 1500, max_depth= 4, min_samples_split= 10,

                                         learning_rate= 0.05, loss='huber').fit(x_train, y_train)
scores = cross_val_score(lasso, x_test, y_test, cv=5)

print("Lasso R-Square : %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(ridge, x_test, y_test, cv=5)

print("Ridge R-Square: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(bayesian, x_test, y_test, cv=5)

print("BayesianRidge R-Square: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(svr, x_test, y_test, cv=5)

print("SupportVectorRegression R-Square: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(gbr, x_test, y_test, cv=5)

print("GradientBoostingRegression R-Square: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
pred1 = lasso.predict(test_features)

pred2 = gbr.predict(test_features)

pred3 = svr.predict(test_features)

pred = (np.exp(pred1) + np.exp(pred2) +  np.exp(pred3)) / 3 

output=pd.DataFrame({'Id':test_ID, 'SalePrice':pred})

output.to_csv('submission.csv', index=False)
output.head()