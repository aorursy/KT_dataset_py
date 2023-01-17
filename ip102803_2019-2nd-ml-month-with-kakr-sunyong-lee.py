import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#bring in the six packs

df_train = pd.read_csv('../input/train.csv')

df_train
#bring in the six packs

df_test = pd.read_csv('../input/test.csv')

df_test
#check the decoration

df_train.columns
#descriptive statistics summary

df_train['price'].describe()
#histogram

sns.distplot(df_train['price']);
#skewness and kurtosis

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
#scatter plot view/saleprice

var = 'view'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,8000000));
#scatter plot grade/saleprice

var = 'grade'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,8000000));
#box plot grade/saleprice

var = 'grade'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=8000000);
var = 'yr_built'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=8000000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price')['price'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
corr_matrix = df_train.corr()

corr_matrix["price"].sort_values(ascending=False)
#scatterplot

sns.set()

cols = ['price', 'sqft_living', 'sqft_above', 'view', 'grade', 'bathrooms', 'sqft_living15']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['price'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#histogram and normal probability plot

sns.distplot(df_train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['price'], plot=plt)
#applying log transformation

df_train['price'] = np.log(df_train['price'])
#transformed histogram and normal probability plot

sns.distplot(df_train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['price'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['grade'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['grade'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['yr_built'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['yr_built'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['bedrooms'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['bedrooms'], plot=plt)
#scatter plot 

var = 'bedrooms'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'bathrooms'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_living'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_lot'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'floors'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'waterfront'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'view'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'condition'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'grade'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_above'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_basement'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'yr_built'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'yr_renovated'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_living15'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'sqft_lot15'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'zipcode'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'lat'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#scatter plot 

var = 'long'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,18));
#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn import metrics



from sklearn.svm import SVC

from xgboost import XGBClassifier



%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
df2_train = pd.read_csv("../input/train.csv")

df2_test = pd.read_csv("../input/test.csv")
feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'lat', 'long', 'sqft_living15', 'sqft_lot15',

                 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']

X_test = df2_test[feature_names]
X_test = pd.get_dummies(X_test)
X_test
feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'lat', 'long', 'sqft_living15', 'sqft_lot15',

                 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']

X_train = df2_train[feature_names]
X_train
X_train = pd.get_dummies(X_train)

X_train 
label_name = 'price'



y_train = df2_train[label_name]



print(y_train.shape)

y_train.head()
print('X.shape: {}, y.shape{}'.format(X_train.shape, y_train.shape))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler

%matplotlib inline

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import RidgeCV
lgb = lgb.LGBMRegressor(objective='regression', num_leaves = 5, n_estimators=720, learning_rate=0.05,

                                   max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319, 

                                   feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)



lgb.fit(X_train, y_train)
prediction8 = lgb.predict(X_test)

print(prediction8.shape)

prediction8[:20]
submission8 = pd.DataFrame({'id':df_test['id'],'price':prediction8})

submission8.head()
filename = 'House Price Prediction_8.csv'

submission8.to_csv(filename,index=False)

print('Saved file: ' + filename)
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

xgb.fit(X_train, y_train)
prediction9 = xgb.predict(X_test)

print(prediction9.shape)

prediction9[:20]
submission9 = pd.DataFrame({'id':df_test['id'],'price':prediction9})

submission9.head()
filename = 'House Price Prediction_9.csv'

submission9.to_csv(filename,index=False)

print('Saved file: ' + filename)
from sklearn.ensemble import GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =4)

GBoost.fit(X_train, y_train)

prediction7 = GBoost.predict(X_test)

print(prediction7.shape)

prediction7[:20]
submission7 = pd.DataFrame({'id':df_test['id'],'price':prediction7})

submission7.head()
filename = 'House Price Prediction_7.csv'

submission7.to_csv(filename,index=False)

print('Saved file: ' + filename)
import numpy as np



from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
X_train.shape
X_test2.shape
y_test2.shape
# 개별 ML 모델을 위한 Classifier 생성.

knn_clf  = KNeighborsRegressor(n_neighbors=4)

rf_clf = RandomForestRegressor(n_estimators=1000, random_state=0)

ada_clf = AdaBoostRegressor(n_estimators=1000)

Gboost = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.05, max_depth=5)
# 최종 Stacking 모델을 위한 Classifier생성. 



GBoost_stacking = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =4)

# 개별 모델들을 학습. 

knn_clf.fit(X_train, y_train)

rf_clf.fit(X_train , y_train)

ada_clf.fit(X_train, y_train)

Gboost.fit(X_train, y_train)
# 학습된 개별 모델들이 각자 반환하는 예측 데이터 셋을 생성

knn_pred = knn_clf.predict(X_train)

rf_pred = rf_clf.predict(X_train)

ada_pred = ada_clf.predict(X_train)

gbm_pred = Gboost.predict(X_train)

pred = np.array([knn_pred, rf_pred, ada_pred, gbm_pred])

print(pred.shape)



# transpose를 이용해 행과 열의 위치 교환. 컬럼 레벨로 각 알고리즘의 예측 결과를 피처로 만듦. 

pred2 = np.transpose(pred)

print(pred2.shape)

print(y_test2.shape)

print(X_test2.shape)
GBoost_stacking.fit(pred2, y_train)
knn_pred_f = knn_clf.predict(X_test)

rf_pred_f = rf_clf.predict(X_test)

ada_pred_f = ada_clf.predict(X_test)

gbm_pred_f = Gboost.predict(X_test)



pred_f = np.array([knn_pred_f, rf_pred_f, ada_pred_f,gbm_pred_f])

pred_f2 = np.transpose(pred_f)
prediction18 = GBoost_stacking.predict(pred_f2)

print(prediction18.shape)

prediction18[:20]
submission18 = pd.DataFrame({'id':df_test['id'],'price':prediction18})

submission18.head()
filename = 'House Price Prediction_18.csv'

submission18.to_csv(filename,index=False)

print('Saved file: ' + filename)