# Loading packages

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from scipy import stats

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.isnull().sum()
test.isnull().sum()
train.nunique()
train['price'].describe()
#histogram

f, ax = plt.subplots(figsize = (8, 6))

sns.distplot(train['price'])
train['price'].skew()
train['price'].kurt()
# Scaling Example

# generate 1000 data points randomly drawn from an exponential distribution

from mlxtend.preprocessing import minmax_scaling

import numpy

original_data = numpy.random.exponential(size = 1000)



# mix-max scale the data between 0 and 1

scaled_data = minmax_scaling(original_data, columns = [0])



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled data")

del numpy #import numpy를 해제
# normalize the exponential data with boxcox

normalized_data = stats.boxcox(original_data)



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(normalized_data[0], ax=ax[1])

ax[1].set_title("Normalized data")
#histogram



fig = plt.figure(figsize = (8, 6))



fig.add_subplot(1, 3, 1)

sns.distplot(sp.sqrt(train['price'])).set_title('square-root') # square root transformation



fig.add_subplot(1, 3, 2)

sns.distplot(train['price'] ** (1/float(4.0))).set_title('fourth-root') # fourth square root transformation



fig.add_subplot(1, 3, 3)

sns.distplot(sp.special.log1p(train['price'])).set_title('log1p') # log1p transformation
# price에 square-root를 취함

data = pd.concat([train['price'], train['sqft_living']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.regplot(x = 'sqft_living', y = 'price', data = data)
# price에 fourth-root를 취함

data = pd.concat([train['price'] ** (1/float(4.0)), train['sqft_living']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.regplot(x = 'sqft_living', y = 'price', data = data)
fig = plt.figure(figsize = (8, 6))



fig.add_subplot(1, 3, 1)

data = pd.concat([pd.DataFrame(sp.sqrt(train['price'])), train['sqft_living']], axis = 1)

data.columns.values[0] = 'price'

reg = sns.regplot(x = 'sqft_living', y = 'price', data = data).set_title('square-root transformation')



fig.add_subplot(1, 3, 2)

data = pd.concat([(train['price'] ** 1/float(4.0)), train['sqft_living']], axis = 1)

reg = sns.regplot(x = 'sqft_living', y = 'price', data = data).set_title('fourth-root transformation')



fig.add_subplot(1, 3, 3)

data = pd.concat([sp.special.log1p(train['price']), train['sqft_living']], axis = 1)

reg = sns.regplot(x = 'sqft_living', y = 'price', data = data).set_title('log1p transformation')
train['price'] = sp.special.log1p(train['price'])
# correlation이 높은 상위 10개의 heatmap

# continuous + sequential variables --> spearman

# abs는 반비례관계도 고려하기 위함

# https://www.kaggle.com/junoindatascience/let-s-eda-it 준호님이 수정해 준 코드로 사용하였습니다. 



cor_abs = abs(train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = sp.array(sp.stats.spearmanr(train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
data = pd.concat([train['price'], train['grade']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.boxplot(x = 'grade', y = 'price', data = data)
data = pd.concat([train['price'], train['sqft_living']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.regplot(x = 'sqft_living', y = 'price', data = data)
data = pd.concat([train['price'], train['sqft_living15']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.regplot(x = 'sqft_living15', y = 'price', data = data)
data = pd.concat([train['price'], train['sqft_above']], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.regplot(x = 'sqft_above', y = 'price', data = data)
data = pd.concat([train['price'], train['bathrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bathrooms', y="price", data=data)
data = pd.concat([train['price'], train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
test[test['bedrooms'] > 10]
data = pd.concat([train['price'], train['floors']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='floors', y="price", data=data)
train.head()
train['sqft_living_gap'] = train['sqft_living15'] - train['sqft_living']

train['sqft_lot_gap'] = train['sqft_lot15'] - train['sqft_lot']
train['sqft_living_gap'].describe()
data = pd.concat([train['price'], train['sqft_living_gap']], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.regplot(x = 'sqft_living_gap', y = 'price', data = data)
train['sqft_lot_gap'].describe()
data = pd.concat([train['price'], train['sqft_lot_gap']], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.regplot(x = 'sqft_lot_gap', y = 'price', data = data)
data = pd.concat([train['price'], train['sqft_living']], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.regplot(x = 'sqft_living', y = 'price', data = data)
sorted(set(test['sqft_living']) - set(train['sqft_living']), reverse = True)[0:10]
max(test['sqft_living'])
train.loc[train['sqft_living'] > 12000]
train = train.loc[(train['id'] != 5108) & (train['id'] != 8912),]
data = pd.concat([train['price'], train['grade']], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x = 'grade', y = 'price', data = data)
train[train['grade'] == 3]
train.loc[(train['price']>14.5) & (train['grade'] == 7)]
train.loc[(train['price']>14.5) & (train['grade'] == 8)]
train.loc[(train['price'] > 15.5) & (train['grade'] == 11)]
train = train.loc[(train['id'] != 12346) & (train['id'] != 7173) & (train['id'] != 2775)]
f, ax = plt.subplots(figsize = (8, 6))

sns.distplot(train['sqft_lot'])
f, ax = plt.subplots(figsize = (8, 6))

sns.distplot(sp.special.log1p(train['sqft_lot']))
data = pd.concat([train['price'], sp.special.log1p(train['sqft_lot'])], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.regplot(x = 'sqft_lot', y = 'price', data = data)
for df in [train,test]:

    df['year'] = df.date.apply(lambda x: x[0:4]).astype(int)

    df['month'] = df.date.apply(lambda x: x[4:6]).astype(int)

    df['day'] = df.date.apply(lambda x: x[6:8]).astype(int)

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: sp.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
for df in [train,test]:

    # bedrooms와 bathrooms의 개수를 종합한 방의 개수

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    

    # grade와 condition의 곱을 통한 새로운 지표 생성

    df['grade_condition'] = df['grade'] * df['condition']

    

    # 면적관련 변수들을 이용한 파생변수 생성

    

    # 15년 변화 이전 부지와 주거공간의 평방피트 합

    df['sqft_total'] = df['sqft_living'] + df['sqft_lot'] 

    

    # 15년 변화 이전 부지, 주거공간, 지하실 제외, 지하실의 평방피트 합

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    

    # 15년 변화 이후 부지와 주거공간의 평방피트 합

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    

    # 주거공간 평방피트의 변화 이전과 변화 이후의 차이

    df['sqft_living_gap'] = sp.absolute(df['sqft_living15'] - df['sqft_living'])

    

    # 15년 변화 이후 부지 평방피트의 차이

    df['sqft_lot_gap'] = sp.absolute(df['sqft_lot15'] - df['sqft_lot'])

    

    # 재건축 여부를 통한 파생변수 생성

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')

    

    df['garret'] = (df.floors%1==0.5).astype(int)

    df['diff_of_rooms'] = sp.absolute(df['bedrooms'] - df['bathrooms'])

    df['living_per_floors'] = df['sqft_living'] / df['floors']

    df['total_score'] = df['condition'] + df['grade'] + df['view']

    df['living_per_lot'] = df['sqft_living'] / df['sqft_lot']

    df['gap_living_per_floor'] = df['sqft_living_gap'] / df['floors']

    df['exist_special'] = df.garret + df.waterfront + df.is_renovated
train['per_price'] = train['price']/train['sqft_total_size']

zipcode_price = train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

train = pd.merge(train,zipcode_price,how='left',on='zipcode')

test = pd.merge(test,zipcode_price,how='left',on='zipcode')



for df in [train,test]:

    df['zipcode_mean'] = df['mean'] * df['sqft_total_size']

    df['zipcode_var'] = df['var'] * df['sqft_total_size']

    del df['mean']; del df['var']

    

del train['per_price']
# transformation

skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    train[c] = sp.special.log1p(train[c].values)

    test[c] = sp.special.log1p(test[c].values)
train.head()
X_features_dummy = pd.get_dummies(train, columns = ['waterfront', 'zipcode', 'yr_built', 'yr_renovated', 'year', 'month', 'day', 'garret', 'is_renovated'])

Y_features_dummy = pd.get_dummies(test, columns = ['waterfront', 'zipcode', 'yr_built', 'yr_renovated', 'year', 'month', 'day', 'garret', 'is_renovated'])
from sklearn.preprocessing import LabelEncoder



le_columns = ['bedrooms', 'bathrooms', 'floors', 'view', 'condition', 'grade', 'total_rooms', 'grade_condition', 'exist_special', 'living_per_lot', 'living_per_floors',

              'gap_living_per_floor', 'total_score', 'diff_of_rooms']





le = LabelEncoder()



for i in le_columns : 

    X_features_dummy[i] = le.fit_transform(X_features_dummy[i])

    Y_features_dummy[i] = le.fit_transform(Y_features_dummy[i])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



test_id = Y_features_dummy['id']

Y_test = Y_features_dummy.drop(['id'], axis = 1, inplace = False)

y_target = X_features_dummy['price']

X_data = X_features_dummy.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.25, random_state = 42)
lr = LinearRegression()

lr.fit(X_train, y_train)
pred = sp.special.expm1(lr.predict(X_test))

y_test = sp.special.expm1(y_test)

mse = mean_squared_error(y_test, pred)

rmse = mse ** float(0.5)

print('RMSE : {0:.3F}'.format(rmse))
import statsmodels.formula.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std



y = X_features_dummy['price']

X = X_features_dummy.drop(['price', 'id'], axis = 1, inplace = False)

tests = Y_features_dummy



# X_features_dummy 데이터 내, 모든 X를 이용하여 Y인 price를 예측하도록 함

result = sm.OLS(y, X).fit()



print(result.summary())
# fitted values

model_fitted_y = result.fittedvalues



# model residuals

model_residuals = result.resid



# normalized residuals

model_norm_residuals = result.get_influence().resid_studentized_internal



# absolute squared normalized residuals

model_norm_residuals_abs_sqrt = sp.sqrt(sp.absolute(model_norm_residuals))



# absolute residuals

model_abs_resid = sp.absolute(model_residuals)



# leverage, from statsmodel internals

model_leverage = result.get_influence().hat_matrix_diag



# cook's distance, from sttasmodels internals

model_cooks = result.get_influence().cooks_distance[0]
plot_lm_1 = plt.figure(1)

plot_lm_1.set_figheight(8)

plot_lm_1.set_figwidth(12)



plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'price', data=X_features_dummy, 

                          lowess=True, 

                          scatter_kws={'alpha': 0.5}, 

                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_1.axes[0].set_xlabel('Fitted values')

plot_lm_1.axes[0].set_ylabel('Residuals')



# annotations

abs_resid = model_abs_resid.sort_values(ascending=False)

abs_resid_top_3 = abs_resid[:3]



for i in abs_resid_top_3.index:

    plot_lm_1.axes[0].annotate(i, 

                               xy=(model_fitted_y[i], 

                                   model_residuals[i]));
from statsmodels.graphics.gofplots import ProbPlot



QQ = ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)



plot_lm_2.set_figheight(8)

plot_lm_2.set_figwidth(12)



plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');



# annotations

abs_norm_resid = sp.flip(sp.argsort(sp.absolute(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]



for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i, 

                               xy=(sp.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
plot_lm_3 = plt.figure(3)

plot_lm_3.set_figheight(8)

plot_lm_3.set_figwidth(12)



plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)

sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_3.axes[0].set_title('Scale-Location')

plot_lm_3.axes[0].set_xlabel('Fitted values')

plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');



# annotations

abs_sq_norm_resid = sp.flip(sp.argsort(model_norm_residuals_abs_sqrt), 0)

abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]



for i in abs_norm_resid_top_3:

    plot_lm_3.axes[0].annotate(i, 

                               xy=(model_fitted_y[i], 

                                   model_norm_residuals_abs_sqrt[i]));

plot_lm_4 = plt.figure(4)

plot_lm_4.set_figheight(8)

plot_lm_4.set_figwidth(12)



plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)

sns.regplot(model_leverage, model_norm_residuals, 

            scatter=False, 

            ci=False, 

            lowess=True,

            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



plot_lm_4.axes[0].set_xlim(0, 0.20)

plot_lm_4.axes[0].set_ylim(-3, 5)

plot_lm_4.axes[0].set_title('Residuals vs Leverage')

plot_lm_4.axes[0].set_xlabel('Leverage')

plot_lm_4.axes[0].set_ylabel('Standardized Residuals')



# annotations

leverage_top_3 = sp.flip(sp.argsort(model_cooks), 0)[:3]



for i in leverage_top_3:

    plot_lm_4.axes[0].annotate(i, 

                               xy=(model_leverage[i], 

                                   model_norm_residuals[i]))

    

# shenanigans for cook's distance contours

def graph(formula, x_range, label=None):

    x = x_range

    y = formula(x)

    plt.plot(x, y, label=label, lw=1, ls='--', color='red')



p = len(result.params) # number of model parameters



graph(lambda x: sp.sqrt((0.5 * p * (1 - x)) / x), 

      sp.linspace(0.001, 0.200, 50), 

      'Cook\'s distance') # 0.5 line

graph(lambda x: sp.sqrt((1 * p * (1 - x)) / x), 

      sp.linspace(0.001, 0.200, 50)) # 1 line

plt.legend(loc='upper right');
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint
test_id = test['id']

Y_test = test.drop(['id'], axis = 1, inplace = False)
y_target = train['price']

X_data = train.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.25, random_state = 42)
forest_reg = RandomForestRegressor()
import numpy as np

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
param_dist = {"max_depth": [7, 11, 15, 18, 21],

              "max_features": sp_randint(1, 31),

              "min_samples_split": sp_randint(2, 21),

              "min_samples_leaf": sp_randint(1, 21),

              "bootstrap": [True, False],

              "random_state": [42]

             }



n_iter_search = 20

random_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist,

                                   n_iter=n_iter_search)
random_search.fit(X_train, y_train)
report(random_search.cv_results_)
from sklearn.metrics import mean_squared_error as mse
pred = sp.special.expm1(random_search.predict(X_test))

y_test = sp.expm1(y_test)

rmse = (mean_squared_error(y_test, pred)) ** float(0.5)

print('RMSE : {0:.3F}'.format(rmse))
test_pred = sp.special.expm1(random_search.predict(Y_test))
submission = pd.DataFrame({'id' : test_id, 'price' : test_pred})
submission.to_csv('rfgridcv.csv', index=False)
from xgboost import XGBRegressor

from sklearn import model_selection
test_id = test['id']

Y_test = test.drop(['id'], axis = 1, inplace = False)
y_target = train['price']

X_data = train.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.25, random_state = 42)

watchlist=[(X_train,y_train),(X_test,y_test)]
# feature missmatch 에러를 해결하기 위함

Y_test = Y_test[X_test.columns]
xgb_model= XGBRegressor(tree_method='gpu_hist',

                        n_estimators=100000,

                        num_round_boost=500,

                        show_stdv=False,

                        feature_selector='greedy',

                        verbosity=0,

                        reg_lambda=10,

                        reg_alpha=0.01,

                        learning_rate=0.001,

                        seed=42,

                        colsample_bytree=0.8,

                        colsample_bylevel=0.8,

                        subsample=0.8,

                        n_jobs=-1,

                        gamma=0.005,

                        base_score=np.mean(y_target)

                        )
xgb_model.fit(X_train,y_train, verbose=False, eval_set=watchlist,

              eval_metric='rmse',

              early_stopping_rounds=1000)
xgb_score=mse(sp.special.expm1(xgb_model.predict(X_test)),sp.special.expm1(y_test))**0.5



print("RMSE : {}".format(xgb_score))
xgb_pred = sp.special.expm1(xgb_model.predict(Y_test))
submission=pd.read_csv('../input/sample_submission.csv')

submission['price']= xgb_pred

submission.to_csv('xgb_submission.csv', index = False)
import lightgbm as lgb
y_target = train['price']

X_data = train.drop(['price', 'id'], axis = 1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.2, random_state = 42)



model_lgb=lgb.LGBMRegressor(

                           learning_rate=0.001,

                           n_estimators=100000,

                           subsample=0.6,

                           colsample_bytree=0.6,

                           reg_alpha=0.2,

                           reg_lambda=10,

                           num_leaves=35,

                           silent=True,

                           min_child_samples=10,

                            

                           )



model_lgb.fit(X_train,y_train,eval_set=(X_test,y_test),verbose=0,early_stopping_rounds=1000,

              eval_metric='rmse')



lgb_score=mse(sp.special.expm1(model_lgb.predict(X_test)),sp.special.expm1(y_test))**0.5

print("RMSE unseen : {}".format(lgb_score))
lgb_pred = sp.special.expm1(model_lgb.predict(Y_test))
submission=pd.read_csv('../input/sample_submission.csv')

submission['price']= lgb_pred

submission.to_csv('lgbm_submission.csv', index = False)
score=lgb_score+xgb_score

lgb_ratio=1-lgb_score/score

xgb_ratio=1-xgb_score/score

predict=lgb_pred*(lgb_ratio)+xgb_pred*(xgb_ratio)

print('lgb_ratio={}, xgb_ratio={}'.format(lgb_ratio,xgb_ratio))

submission=pd.read_csv('../input/sample_submission.csv')

submission.loc[:,'price']=predict

submission.to_csv('xgb_lgbm.csv',index=False)