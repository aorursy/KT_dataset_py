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
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train # 0 to 1351
test_id #1352 to1689
test_id = test.copy()
train.shape
test.shape
train.drop(['house-id'], inplace = True, axis = 1)

test.drop(['house-id'], inplace = True, axis = 1)
train.drop(['data-url'], inplace = True, axis = 1)

test.drop(['data-url'], inplace = True, axis = 1)
corr_overall = train.corr()

corr_overall
fig, plot = plt.subplots()

plot.scatter(x= train['buildingSize'], y= train['data-price'])

plt.ylabel('data-price', fontsize=13)

plt.xlabel('buildingSize', fontsize=13)

plt.show()
#train = train.drop(train[(train['buildingSize']>800) & (train['data-price']>6000000)].index) #drop rows that meet this criteria
fig, plot = plt.subplots()

plot.scatter(x= train['buildingSize'], y= train['data-price'])

plt.ylabel('data-price', fontsize=13)

plt.xlabel('buildingSize', fontsize=13)

plt.show()
train.shape
test_A = test
train_A = train.drop(['data-price'], axis=1)

y = pd.DataFrame(train['data-price'])

features = pd.concat([train_A, test_A]).reset_index(drop=True)
sns.distplot(y['data-price'], color = 'b',kde = True)

plt.title('Distribution of data-price')

plt.xlabel('data-price')

plt.ylabel('Number of occurences')

    

#skewness

skewness = y['data-price'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

#kurtosis    

print('The kurtosis is ' + str(y['data-price'].kurt()))
y_log = pd.DataFrame(np.log1p(y['data-price']))
sns.distplot(y_log['data-price'], color = 'b',kde = True)

plt.title('Distribution of data-price')

plt.xlabel('data-price')

plt.ylabel('Number of occurences')

    

#skewness

skewness = y_log['data-price'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

#kurtosis    

print('The kurtosis is ' + str(y_log['data-price'].kurt()))
features.head()
features.dtypes
features.drop(['bedroom'],inplace = True, axis = 1)
import datetime as dt

features['data-date'] = pd.to_datetime(features['data-date'])

features['data-date']=features['data-date'].map(dt.datetime.toordinal)
features.head()
features.dtypes
features.drop(['area'], inplace = True, axis = 1)
norm_cols = ['data-location','type']

dumies=pd.get_dummies(features[norm_cols],drop_first=True)

dumies.head()
for column in norm_cols:

    features.drop([column],axis=1,inplace=True)
features.head()
features=pd.concat([ features,dumies],axis=1)

features.head()
features.dtypes
features_na = (features.isnull().sum() / len(features)) * 100

#drop features without missing values

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :features_na})



#plot

f, ax = plt.subplots(figsize=(10, 8))

sns.barplot(x=features_na.index, y=features_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent', fontsize=15)

plt.xticks(rotation='90')

plt.title('Percent of missing data by feature', fontsize=15)

features['garage']
plt.hist(features['garage'])

plt.show()



#skewness

skewness = features['garage'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['garage'].kurt()))
features['garage'] = np.log1p(features['garage'])
plt.hist(features['garage'])

plt.show()



#skewness

skewness = features['garage'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['garage'].kurt()))
features['garage'].fillna(features['garage'].mean(), inplace = True)
plt.hist(features['buildingSize'])

plt.show()



#skewness

skewness = features['buildingSize'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['buildingSize'].kurt()))
features['buildingSize'] = np.log1p(features['buildingSize'])
plt.hist(features['buildingSize'])

plt.show()



#skewness

skewness = features['buildingSize'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['buildingSize'].kurt()))
features['buildingSize'].fillna(features['buildingSize'].mean(), inplace = True)
plt.hist(features['erfSize'])

plt.show()



#skewness

skewness = features['erfSize'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['erfSize'].kurt()))
features['erfSize'] = np.log1p(features['erfSize'])
plt.hist(features['erfSize'])

plt.show()



#skewness

skewness = features['erfSize'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['erfSize'].kurt()))
features['erfSize'].fillna(features['erfSize'].mean(), inplace = True)
'''plt.hist(features['bedroom'])

plt.show()



#skewness

skewness = features['bedroom'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['bedroom'].kurt()))'''
#features['bedroom'] = np.log1p(features['bedroom'])
'''plt.hist(features['bedroom'])

plt.show()



#skewness

skewness = features['bedroom'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['bedroom'].kurt()))'''
#features['bedroom'].fillna(features['bedroom'].mean(), inplace = True)
plt.hist(features['bathroom'])

plt.show()



#skewness

skewness = features['bathroom'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['bathroom'].kurt()))
features['bathroom'] = np.log1p(features['bathroom'])
plt.hist(features['bathroom'])

plt.show()



#skewness

skewness = features['bathroom'].skew()

if (skewness > -0.5) & (skewness < 0.5):

    print('The data is fairly symmetrical with skewness of ' + str(skewness))

elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):

    print('The data is moderately skewed with skewness of ' + str(skewness))

elif (skewness < -1) | (skewness > 1):

    print('The data is highly skewed with skewness of ' + str(skewness))

        

#kurtosis

print('The kurtosis is ' + str(features['bathroom'].kurt()))
features['bathroom'].fillna(features['bathroom'].mean(), inplace = True)
features.head()
new_train = features.iloc[:1351,:]

new_test = features.iloc[1351:,:]
X_train = new_train

X_test = new_test
y_train = y_log
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_train)
X_standardize = pd.DataFrame(X_scaled, columns = X_train.columns)

X_standardize.head()
X_standardize.describe().loc['std']
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)

lin_pred = np.expm1(y_pred)
lin_sub = pd.DataFrame(lin_pred)
lin_sub.columns = ['price']
lin_sub.shape
lin_sub.head()
lin_sub['house-id'] = test_id['house-id'].astype(int)
lin_sub.head()
lin_sub.set_index('house-id', inplace = True)
lin_sub.to_csv('Linear_regression.csv')
data = pd.read_csv('Linear_regression.csv')
data.head()
from sklearn.linear_model import Ridge

from sklearn import metrics



ridge = Ridge()

ridge.fit(X_train, y_train)



y_ridge = ridge.predict(X_test)

r_ridge = np.expm1(y_ridge)

ridge_sub = pd.DataFrame(r_ridge)



ridge_sub.columns = ['price']

ridge_sub['house-id'] = test_id['house-id'].astype(int)

ridge_sub.set_index('house-id', inplace = True)

ridge_sub.to_csv('Ridge_regression.csv')
ridge_data = pd.read_csv('Ridge_regression.csv')
ridge_data.head()
from sklearn.linear_model import Lasso,ElasticNet
lasso = Lasso(alpha=0.0005)

lasso.fit(X_train, y_train)



y_lasso_pred = lasso.predict(X_test)

lass_pred = np.expm1(y_lasso_pred)

lasso_sub = pd.DataFrame(lass_pred)



lasso_sub.columns = ['price']

lasso_sub['house-id'] = test_id['house-id'].astype(int)

lasso_sub.set_index('house-id', inplace = True)

lasso_sub.to_csv('Lasso_regression.csv')
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



ENet.fit(X_train, y_train)

ENet_pred = ENet.predict(X_test)

net_pred = np.expm1(ENet_pred)

ENet_sub = pd.DataFrame(net_pred)



ENet_sub.columns = ['price']

ENet_sub['house-id'] = test_id['house-id'].astype(int)

ENet_sub.set_index('house-id', inplace = True)

ENet_sub.to_csv('ENet_regression.csv')

dt = pd.read_csv('ENet_regression.csv')

dt.head()
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=300, learning_rate=0.08, gamma=0, subsample=0.8,

                           colsample_bytree=1, max_depth=7)



xgb.fit(X_train,y_train)



predictions = xgb.predict(X_test)

x_pred = np.expm1(predictions)

xgb_sub = pd.DataFrame(x_pred)



xgb_sub.columns = ['price']

xgb_sub['house-id'] = test_id['house-id'].astype(int)

xgb_sub.set_index('house-id', inplace = True)

xgb_sub.to_csv('xgb_regression.csv')
dat = pd.read_csv('xgb_regression.csv')

dat.head()
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet, regressor, lasso,ridge))
averaged_models.fit(X_train, y_train)

stacked_pred = np.expm1(averaged_models.predict(X_test))



st_sub = pd.DataFrame(stacked_pred)



st_sub.columns = ['price']

st_sub['house-id'] = test_id['house-id'].astype(int)

st_sub.set_index('house-id', inplace = True)

st_sub.to_csv('stacked_regression.csv')
data = pd.read_csv('stacked_regression.csv')

data.head()