import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
df.info()
test_df.info()
df.isnull().sum()
#visualise null values

sns.heatmap(df.isnull(), yticklabels=False, cbar='False')
sns.heatmap(test_df.isnull(), yticklabels=False, cbar='False')
#outlier removal

fig, ax = plt.subplots()

ax.scatter(x = df['GrLivArea'], y = df['SalePrice'], c='blue')

plt.ylabel('SalePrice', fontsize=9)

plt.xlabel('GrLivArea', fontsize=9)

plt.show()
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter(df['GrLivArea'], df['SalePrice'], c='blue')

plt.ylabel('SalePrice', fontsize=9)

plt.xlabel('GrLivArea', fontsize=9)

plt.show()
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)

df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)

df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)

df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)

df['GrLivArea'].fillna(df['GrLivArea'].mean(), inplace=True)

df['SalePrice'].fillna(df['SalePrice'].mean(), inplace=True)

df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)

df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)

df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)

df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)

df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0], inplace=True)

df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)

df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)

df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)

df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True) #drop, too many null values
#repeat for test_df

test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean(), inplace=True)

test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0], inplace=True)

test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean(), inplace=True)

test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0], inplace=True)

test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0], inplace=True)

test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0], inplace=True)

test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0], inplace=True)

test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0], inplace=True)

test_df['Electrical'].fillna(test_df['Electrical'].mode()[0], inplace=True)

test_df['GarageType'].fillna(test_df['GarageType'].mode()[0], inplace=True)

test_df['GarageYrBlt'].fillna(test_df['GarageYrBlt'].mode()[0], inplace=True)

test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0], inplace=True)

test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0], inplace=True)

test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0], inplace=True)

test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0], inplace=True)

test_df['Utilities'].fillna(test_df['Utilities'].mode()[0], inplace=True)

test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0], inplace=True)

test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0], inplace=True)

test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean(), inplace=True)

test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean(), inplace=True)

test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean(), inplace=True)

test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean(), inplace=True)

test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean(), inplace=True)

test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mean(), inplace=True)

test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0], inplace=True)

test_df['GarageCars'].fillna(test_df['GarageCars'].mean(), inplace=True)

test_df['Functional'].fillna(test_df['Functional'].mode()[0], inplace=True)

test_df['GarageArea'].fillna(test_df['GarageArea'].mean(), inplace=True)

test_df['SaleType'].fillna(test_df['SaleType'].mode()[0], inplace=True)

test_df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True) #drop, too many null values
sns.heatmap(test_df.isnull(), yticklabels=False, cbar='False')
sns.heatmap(df.isnull(), yticklabels=False, cbar='False')
#columns with categorical values

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

         'SaleCondition','ExterCond','ExterQual','Foundation','BsmtQual',

         'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',    

         'CentralAir', 'Electrical','KitchenQual','Functional','GarageType',

         'GarageFinish','GarageQual','GarageCond','PavedDrive','LandSlope','Neighborhood']



len(columns)
df.shape
#concatenate train and test df's for making dummy variables for categorical variable

final_df = pd.concat([df,test_df],axis=0)
final_df.shape
final_df = pd.get_dummies(final_df, columns=columns, drop_first=True)

final_df
final_df.shape
#drop duplicates

final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df.drop_duplicates(inplace=True)
final_df.shape
#separate test and train dfs

df=final_df.iloc[:1458,:]

test_df=final_df.iloc[1458:,:]

test_df.drop(['SalePrice'],axis=1,inplace=True)
test_df.shape
df.shape
#remove outliers

from scipy import stats

df[(np.abs(stats.zscore(df)) <3).all(axis=1)]
#selecting correlated features reduced performance of my models so decided not to do it
#correlation matrix

corrmat = df.corr()

corrmat
#find features correlated to Sale Price

def getCorrelatedFeature(corrdata, threshold):

    feature = []

    value = []

    

    for i, index in enumerate(corrdata.index):

        if abs(corrdata[index])> threshold:

            feature.append(index)

            value.append(corrdata[index])

            

    corrdf = pd.DataFrame(data = value, index = feature, columns=['corr value'])

    return corrdf
threshold = 0.5

corr_value = getCorrelatedFeature(corrmat['SalePrice'], threshold)

corr_value.sort_values(by=['corr value'], ascending = False)

len(corr_value)
correlated_data = df[corr_value.index]

correlated_data.head()
fig, ax = plt.subplots(figsize=(15,10))

sns.heatmap(correlated_data.corr(), annot=True, annot_kws={'size': 5})
corr_value = getCorrelatedFeature(corrmat['SalePrice'], threshold)

corr_value
cols = correlated_data.columns

cols
sns.distplot(df['SalePrice'])

sns.distplot(df.skew(),color='blue',axlabel ='Skewness')
X = df.drop(labels=['SalePrice'], axis = 1)

y = df['SalePrice']





#log transform

y= np.log1p(y)
#split df into train and test 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV

import joblib

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
def print_results(results):

    print('BEST PARAMS: {}\\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_train, y_train)
#feature scaling for ridge

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_scaled = sc.transform(X_train)
from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(X_scaled, y_train)

ridge.score(X_scaled, y_train)
parameters = {

    'solver': ['cholesky', 'auto'],

    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]

}



cv = GridSearchCV(ridge, parameters, cv=5)

cv.fit(X_train, y_train)

print_results(cv)



print(cv.best_estimator_)

joblib.dump(cv.best_estimator_, '../../../ridge_model.pkl')
from sklearn.linear_model import Lasso

lasso = Lasso()



parameters = {

    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]

}



cv = GridSearchCV(lasso, parameters, cv=5)

cv.fit(X_train, y_train)

print_results(cv)

print('best estimator:', cv.best_estimator_)

joblib.dump(cv.best_estimator_, '../../../lasso_model.pkl')
from sklearn.linear_model import ElasticNet

en = ElasticNet(l1_ratio=0.5)



parameters = {

    'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]

}



cv = GridSearchCV(en, parameters, cv=5)

cv.fit(X_train, y_train)

print_results(cv)

print('best estimator:', cv.best_estimator_)

joblib.dump(cv.best_estimator_, '../../../en_model.pkl')
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_depth = 2, min_samples_leaf=10)

dtr.fit(X_scaled, y_train)

dtr.score(X_scaled, y_train)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1)

gbr.fit(X_train, y_train)

gbr.score(X_train, y_train)
gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=120, random_state=42)

gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
import xgboost

xgb_reg = xgboost.XGBRegressor(n_estimators = 550, random_state=42)

xgb_reg.fit(X_train, y_train)

print(xgb_reg.score(X_train, y_train))
##importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model##

from xgboost  import plot_importance

plt.rcParams['figure.figsize'] = [15, 25]

plt.rcParams['figure.dpi'] = 200

plot_importance(xgb_reg)

plt.show()
y_pred=xgb_reg.predict(test_df)

y_pred
#inverse log transform for real prediction values

y_pred[y_pred<0] = 0

y_pred = np.expm1(y_pred)

y_pred
#make submission df

prediction = pd.DataFrame(y_pred)

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

prediction_df = pd.concat([submission['Id'], prediction], axis=1)

prediction_df.columns=['Id','SalePrice']

prediction_df.to_csv('sample_submission.csv',index=False)