import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, matplotlib, math, copy

import seaborn as sns

import matplotlib.pyplot as plt



print(os.listdir("../input"))

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

merchants = pd.read_csv('../input/merchants.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

historical_transactions = pd.read_csv('../input/historical_transactions.csv')

Data_Dictionary = pd.read_excel('../input/Data_Dictionary.xlsx')
def tf(pd_series):

    return np.array(pd_series).reshape(-1,1)



def transform_viz(pd_series,transform_list = [QuantileTransformer(), MinMaxScaler(), StandardScaler(),PowerTransformer()]):

    transform_list = [QuantileTransformer(), MinMaxScaler(), StandardScaler(),PowerTransformer()]

    fig, ax = plt.subplots(1,len(transform_list)+1,figsize = (18,3))

    sns.distplot(pd_series.dropna().sample(1000),ax = ax[0]);

    for i, transformer in enumerate(transform_list):

        sns.distplot(transformer.fit_transform(tf(pd_series.dropna().sample(1000))),ax = ax[i+1]);



def plot_details(pd_series):

    plt.figure(figsize = (3,3))

    std, mean, skew = pd_series.std(), pd_series.mean(), pd_series.skew()

    print('shape: {}, mean: {:.2f}, std: {:.2f}, skew: {:.2f}'.format(pd_series.shape,mean,std,skew))

    sns.distplot(pd_series)

    plt.show();

    return std, mean, skew



def remove_outliers_and_skew(pd_series,skew_threshold = 1, remove_outliers = True, deviations = 2.15):

    """

    args:

        pd-series: required

        skew_threshold: skew calc above which series with be log-transformed

        remove_outliers: will return pd_series with ...

        deviations: ...

    """

    print('original distribution')

    std, mean, skew = plot_details(pd_series)

    if skew > 1:

        print('log transform original')

        new_series = np.log1p(pd_series)

        if remove_outliers:

            std, mean, skew = plot_details(new_series)

            print('update size')

            new_series = new_series[new_series.between(mean - (std * deviations),mean + (std * deviations))]

    else:

        if remove_outliers:

            print('update size')

            new_series = new_series[new_series.between(mean - (std * deviations),mean + (std * deviations))]

        else:

            new_series = pd_series

    plot_details(new_series)

    return new_series
sample_submission.head()
sns.distplot(train['target']);
print(new_merchant_transactions.shape)

new_merchant_transactions.head(2)
print(train.shape)

train.head(2)
train.nunique()
print(test.shape)

test.head(2)
print(merchants.shape)

merchants.head(2)

#merchants.nunique()
print(historical_transactions.shape)

historical_transactions.head(2)
print(historical_transactions[['card_id','merchant_id']].nunique())

print(train.shape[0])

print(test.shape[0])

print((train.shape[0])+(test.shape[0]))
Data_Dictionary
historical_trans_sample = historical_transactions[:100000].nunique()

data_types = historical_transactions.dtypes

features = pd.DataFrame({'unique':historical_trans_sample,'dtypes':data_types})

features
features_numeric = features[features['dtypes']!=object].index.tolist()

print(features_numeric)
features_dummy = features[(features['dtypes']==object)&(features['unique']<30)].index.tolist()

features_dummy
historical_transactions[features_dummy].sample(100000).nunique()
#note that this just pulls a sample from each

columns = 3

rows = math.ceil(len(features_numeric)/columns)



fig, ax = plt.subplots(rows,columns,figsize = (16,8))

for i, feature in list(enumerate(features_numeric)):

    sns.distplot(historical_transactions[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns],axlabel=feature)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
for feature in ['category_2','state_id']:

    features_numeric.remove(feature)

    features_dummy.append(feature)

    

print('numeric features: {}'.format(features_numeric))

print('dummy features: {}'.format(features_dummy))
historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min()

purchase_amount_sum = historical_transactions[['purchase_amount','card_id']].groupby('card_id').sum()['purchase_amount']

purchase_amount_sum_transformed = remove_outliers_and_skew(purchase_amount_sum,1,False,6) ##########

del purchase_amount_sum
sns.distplot(historical_transactions['month_lag'].abs().sample(20000))
historical_transactions['month_lag'] = historical_transactions['month_lag'].abs()

month_lag_abs_mean = historical_transactions[['month_lag','card_id']].groupby('card_id').mean()['month_lag'] ##########

month_lag_abs_mean = remove_outliers_and_skew(month_lag_abs_mean,1,False,6)
dummy_data = historical_transactions[['card_id']].drop_duplicates()

print(dummy_data.shape)

dummy_data.set_index('card_id', inplace = True)

#dummy_data.head()
# Creating dummies of historical transactions from state_id is too large, 29m x 15

# Keeping only those with higher counts

state_id_counts = historical_transactions[['state_id','card_id']].groupby('state_id').count() / historical_transactions.shape[0]
state_id_counts
state_id_counts[state_id_counts['card_id']>.03]
null_states = list(state_id_counts[state_id_counts['card_id']<=.03].index)
#before

historical_transactions[['state_id','card_id']].groupby('state_id').count().head()
#historical_transactions['state_id'].isin(null_states),0

#df.loc[df.my_channel > 20000, 'my_channel'] = 0

historical_transactions.state_id[historical_transactions['state_id'].isin(null_states)] = 0
#after

historical_transactions[['state_id','card_id']].groupby('state_id').count().head()
#after

historical_transactions['state_id'].nunique()
features_dummy
for feature in features_dummy:

    dummies = pd.get_dummies(historical_transactions[feature].astype(str),prefix=feature)

    dummies = pd.merge(dummies,historical_transactions[['card_id']],left_index = True, right_index = True)

    dummies = dummies.groupby('card_id').sum()

    dummy_data = pd.merge(dummy_data, dummies, left_index = True, right_index = True, how = 'inner')

    print(dummy_data.columns)



del dummies    

dummy_data.head(3)
card_data = pd.merge(month_lag_abs_mean.to_frame(), purchase_amount_sum_transformed.to_frame(),left_index = True, right_index = True,how = 'inner')

print(card_data.shape)

card_data.head(2)
print(card_data.shape)

card_data = pd.merge(card_data,dummy_data, left_index = True, right_index = True, how = 'inner')

print(card_data.shape)

del dummy_data

card_data.head(2)
columns = 2



train_numeric = list(train.dtypes[train.dtypes != object].index)

rows = math.ceil(len(train_numeric)/columns)



fig, ax = plt.subplots(rows,columns,figsize = (8,4))

for i, feature in list(enumerate(train_numeric)):

    #sns.distplot(train[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns],axlabel=feature)

    sns.distplot(train[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
results = pd.concat([train,test])

results = results.drop(['first_active_month'],axis = 1)

print(results.shape)

results.head()
card_data.shape
results[results['target'].isnull()].shape[0] + results.dropna().shape[0] - results.shape[0]
all_data = pd.merge(card_data, results, left_index = True, right_on = 'card_id', how = 'inner')

all_data.set_index('card_id', inplace = True)
results_train = all_data.dropna()

results_train.shape
results_test = all_data[all_data['target'].isnull()]

results_test.shape
results_train.shape[0] + results_test.shape[0]
results_test.head(2)
results_test = results_test.drop('target',axis = 1)
X, y = results_train.drop(['target'],axis = 1), results_train['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#del card_data

#del results_train

#del results
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.svm import LinearSVC



from sklearn.linear_model import Ridge

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor
def best_score(model, param_grid):

    grid = GridSearchCV(model, param_grid=param_grid,scoring='neg_mean_squared_error', cv=5)

    grid.fit(X = X_train,y = y_train)

    print(grid.best_score_, grid.best_params_)

    return grid.best_score_, grid.best_params_
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from xgboost import XGBRegressor

#from xgboost import XGBClassifier
#best_score(XGBRegressor(), {'max_depth': [3,6],'n_estimators':[50,300],'learning_rate':[0.05]})

best_score(XGBRegressor(), {'max_depth': [3],'n_estimators':[300],'learning_rate':[0.05]})
grid = GridSearchCV(Ridge(), param_grid={'alpha': [1, 10]},scoring='neg_mean_squared_error', cv=5)

grid.fit(X = X_train,y = y_train)

grid.best_score_
best_score(Ridge(), {'alpha': [1,10]})
best_score(Lasso(), {'alpha': [1,2,4,6,10,20,30,100]})

best_score(Lasso(), {'alpha': [1,2,10, 30,100]})
#best_score(ElasticNet(), {'alpha': [1,2,4,6,10,20,30,100],'l1_ratio':[.2, .5, .8]})

best_score(ElasticNet(), {'alpha': [1],'l1_ratio':[.2]})
model = XGBRegressor(max_depth = 3, n_estimators = 300,learning_rate = 0.05)
#model.fit(X_train,y_train)

model.fit(X,y)

#mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)
sample_submission.head(2)
results_test.head()
predictions = model.predict(results_test)

sub = pd.DataFrame({'card_id':results_test.index, 'target':predictions})

sub.head()
sns.distplot(sub['target'])
sns.distplot(train['target'])
sub.to_csv('submission.csv',index=False)