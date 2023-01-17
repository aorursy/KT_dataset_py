# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
'''

from zipfile import ZipFile



zip_file = ZipFile('../#Data/demand-forecasting-kernels-only.zip')

dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename), parse_dates=['date'])

       for text_file in zip_file.infolist()

       if text_file.filename.endswith('.csv') and not text_file.filename.endswith('submission.csv')}



train = dfs['train.csv']

test = dfs['test.csv']

'''
train = pd.read_csv('../input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])

test = pd.read_csv('../input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
def encode_dates(train):

    train['year'] = train['date'].dt.year.astype('int64')

    train['month'] = train['date'].dt.month.astype('uint8')

    train['day'] = train['date'].dt.day.astype('uint8')

    train['weekday'] = train['date'].dt.dayofweek.astype('uint8')
encode_dates(train)

encode_dates(test)



train.info()

train['weekday'].unique()
train.sort_values(by='date', inplace=True)



test.set_index('id', inplace=True)

test.sort_values(by='date', inplace=True)



 

def split(train):

    model_train = train[train['year'] < 2016].set_index(['date'])

    model_val = train[train['year'] >= 2016].set_index(['date'])

    

    X_train = model_train.drop('sales', axis=1)

    X_test = model_val.drop('sales', axis=1)

    y_train = model_train['sales']

    y_test = model_val['sales']

    

    return X_train, X_test, y_train, y_test
def smape(A, F):

        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
def eval_model(X_train, X_test, y_train, y_test, reg, mode='test'):

    y_pred = reg.predict(X_test)

    y_self_pred = reg.predict(X_train)



    test_compare = X_test.copy()

    test_compare['test'] = y_test

    test_compare['pred'] = y_pred.astype('int')



    train_compare = X_train.copy()

    train_compare['test'] = y_train

    train_compare['pred'] = y_self_pred.astype('int')

    

    #whole_compare = train_compare.append(test_compare)

    

    plt.figure(figsize=(24,6))

    if mode == 'train':

        sns.lineplot(data=train_compare[['year', 'test', 'pred']])

    elif mode == 'test':

        sns.lineplot(data=test_compare[['year', 'test', 'pred']])

    elif mode == 'whole':

        sns.lineplot(data=train_compare[['year', 'test', 'pred']]), sns.lineplot(data=test_compare[['year', 'test', 'pred']])

    

    display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top: 12.580)'.format(smape(y_test, y_pred), smape(y_train, y_self_pred)))
sample = train[train['store'] == 1].reset_index().groupby('date')['sales'].sum().reset_index()

encode_dates(sample)

sample.head()
X_train, X_test, y_train, y_test = split(sample)
'''

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(max_depth=10, random_state=273, n_estimators=500)

forest_reg.fit(X_train, y_train)

'''
'''

eval_model(X_train, X_test, y_train, y_test, forest_reg, 'train')

'''
'''

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2)

poly_X_train = poly.fit_transform(X_train)

poly_X_test = poly.transform(X_test)



#linear_reg = Lasso()

linear_reg = Ridge()

linear_reg.fit(poly_X_train, y_train)



y_pred = linear_reg.predict(poly_X_test)

y_self_pred = linear_reg.predict(poly_X_train)



test_compare = X_test.copy()

test_compare['test'] = y_test

test_compare['pred'] = y_pred.astype('int')



train_compare = X_train.copy()

train_compare['test'] = y_train

train_compare['pred'] = y_self_pred.astype('int')



mode = 'train'



plt.figure(figsize=(24,6))

if mode == 'train':

    sns.lineplot(data=train_compare[['year', 'test', 'pred']])

elif mode == 'test':

    sns.lineplot(data=test_compare[['year', 'test', 'pred']])

elif mode == 'whole':

    sns.lineplot(data=train_compare[['year', 'test', 'pred']]), sns.lineplot(data=test_compare[['year', 'test', 'pred']])



display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top: 12.580)'.format(smape(y_test, y_pred), smape(y_train, y_self_pred)))

'''
'''

rdf = pd.DataFrame({'a':np.arange(-2,5,0.01)})

rdf = rdf.append(rdf, ignore_index=True)

rdf = rdf.assign(rb=lambda x: np.exp(-1/0.5*(x.a-1)**2))

rdf, sns.lineplot(data=rdf, x=rdf['a'], y=rdf['rb'])

'''
def rbf_features(X_train, X_test, gamma=1/0.5):

    rb_X_train = X_train

    rb_X_test = X_test

    for i in range(len(train['weekday'].unique())):

        rb_X_train['rbf'+i] = rb_X_train.apply(lambda x: np.exp(-gamma*(x[colNames[3]]-i)**2))

        #rb_X_train = rb_X_train.assign('rbd'=lambda x: np.exp(-gamma*(x.weekday-i)**2))

        #rb_X_test = rb_X_test.assign(int(i)=lambda x: np.exp(-gamma*(x.weekday-i)**2))

        

    return rb_X_train, rb_X_test
#rbf_features(X_train, X_test)

#rb_X_train
'''

rb_X_train = X_train

rb_X_test = X_test



gamma = 1/0.5



rb_X_train = rb_X_train.assign(rb0=lambda x: np.exp(-gamma*(x.weekday-0)**2))

rb_X_train = rb_X_train.assign(rb1=lambda x: np.exp(-gamma*(x.weekday-1)**2))

rb_X_train = rb_X_train.assign(rb2=lambda x: np.exp(-gamma*(x.weekday-2)**2))

rb_X_train = rb_X_train.assign(rb3=lambda x: np.exp(-gamma*(x.weekday-3)**2))

rb_X_train = rb_X_train.assign(rb4=lambda x: np.exp(-gamma*(x.weekday-4)**2))

rb_X_train = rb_X_train.assign(rb5=lambda x: np.exp(-gamma*(x.weekday-5)**2))

rb_X_train = rb_X_train.assign(rb6=lambda x: np.exp(-gamma*(x.weekday-6)**2))



rb_X_test = rb_X_test.assign(rb0=lambda x: np.exp(-gamma*(x.weekday-0)**2))

rb_X_test = rb_X_test.assign(rb1=lambda x: np.exp(-gamma*(x.weekday-1)**2))

rb_X_test = rb_X_test.assign(rb2=lambda x: np.exp(-gamma*(x.weekday-2)**2))

rb_X_test = rb_X_test.assign(rb3=lambda x: np.exp(-gamma*(x.weekday-3)**2))

rb_X_test = rb_X_test.assign(rb4=lambda x: np.exp(-gamma*(x.weekday-4)**2))

rb_X_test = rb_X_test.assign(rb5=lambda x: np.exp(-gamma*(x.weekday-5)**2))

rb_X_test = rb_X_test.assign(rb6=lambda x: np.exp(-gamma*(x.weekday-6)**2))

'''

#rb_X_train, rb_X_test

'''

rb_X_train.head()

'''
'''

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler



X_train, X_test, y_train, y_test = split(sample)



#poly_X_train = rb_X_train.drop(['month', 'day', 'weekday', 'year'], axis=1)

#poly_X_test = rb_X_test.drop(['month', 'day', 'weekday', 'year'], axis=1)



#scaler = StandardScaler()

#poly_X_train = poly.fit_transform(poly_X_train)

#poly_X_test = poly.transform(poly_X_test)



poly = PolynomialFeatures(degree=2)

poly_X_train = poly.fit_transform(X_train)

poly_X_test = poly.transform(X_test)

#poly_X_train = poly.fit_transform(rb_X_train)

#poly_X_test = poly.transform(rb_X_test)





#poly_X_train = rb_X_train.drop(['month', 'day', 'weekday', 'year'], axis=1)

#poly_X_test = rb_X_test.drop(['month', 'day', 'weekday', 'year'], axis=1)



#linear_reg = Lasso(alpha=10)

linear_reg = Ridge(alpha=10)#alpha = 1000)

linear_reg.fit(poly_X_train, y_train)



y_pred = linear_reg.predict(poly_X_test)

y_self_pred = linear_reg.predict(poly_X_train)



test_compare = X_test.copy()

test_compare['test'] = y_test

test_compare['pred'] = y_pred.astype('int')



train_compare = X_train.copy()

train_compare['test'] = y_train

train_compare['pred'] = y_self_pred.astype('int')



mode = 'train'



plt.figure(figsize=(24,6))

if mode == 'train':

    sns.lineplot(data=train_compare[['year', 'test', 'pred']])

elif mode == 'test':

    sns.lineplot(data=test_compare[['year', 'test', 'pred']])

elif mode == 'whole':

    sns.lineplot(data=train_compare[['year', 'test', 'pred']]), sns.lineplot(data=test_compare[['year', 'test', 'pred']])



display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top: 12.580)'.format(smape(y_test, y_pred), smape(y_train, y_self_pred)))

'''
'''

poly_X_train.head(20)

'''
'''

sample2 = train[train['item'] == 1].groupby(['date', 'store'])['sales'].sum() #.reset_index()

#sample2 = train[train['store'] == 1].groupby(['date', 'item'])['sales'].sum().reset_index()

#encode_dates(sample2)

sample2#.head()



# demand for same item is +/- the same fpr all the shops, while demand at same shop varies significantly for different items 

#=> split set by items and encode shops

'''
'''

items = np.sort(train['item'].unique())



for i in [1]:#items:

    sub_sample = train[train['item'] == i]

    X_train, X_test, y_train, y_test = split(sub_sample)

    label encode for dates

    poly transform for dates

    one hot encode for shops

    

    linear_reg = Ridge(alpha=10)#alpha = 1000)

    linear_reg.fit(poly_X_train, y_train)

    X_train['pred'] = linear_reg.predict(X_train)

    X_test['pred'] = linear_reg.predict(X_test)

    

    display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top total: 12.580)'

    .format(smape(y_test, X_test['pred']), smape(y_train, X_train['pred'])))

    '''
'''

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import PolynomialFeatures



a = train[train['item'] == 1]



ct = ColumnTransformer([("onehot", OneHotEncoder(), ['store']),

                        ("poly", PolynomialFeatures(degree=2), ['year', 'month', 'day', 'weekday'])])

ct.fit_transform(a)

'''

'''

sample = train[train['store'] == 1].reset_index().groupby('date')['sales'].sum().reset_index()

sample2 = train[train['item'] == 1]

sample3 = train[(train['item'] == 2) & (train['store'] == 1)]



sample3.head()

'''
'''

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures



X_train, X_test, y_train, y_test = split(sample3)



ct = ColumnTransformer([#("onehot", OneHotEncoder(categories='auto'), ['store']),

                        ("poly", PolynomialFeatures(degree=2), ['year', 'month', 'day', 'weekday'])])

poly_X_train = ct.fit_transform(X_train)

poly_X_test = ct.transform(X_test)



#linear_reg = Lasso(alpha=10)

linear_reg = Ridge(alpha=10)#alpha = 1000)

linear_reg.fit(poly_X_train, y_train)



y_pred = linear_reg.predict(poly_X_test)

y_self_pred = linear_reg.predict(poly_X_train)



test_compare = X_test.copy()

test_compare['test'] = y_test

test_compare['pred'] = y_pred.astype('int')



train_compare = X_train.copy()

train_compare['test'] = y_train

train_compare['pred'] = y_self_pred.astype('int')



mode = 'train'



plt.figure(figsize=(24,6))

if mode == 'train':

    sns.lineplot(data=train_compare[['test', 'pred']])

elif mode == 'test':

    sns.lineplot(data=test_compare[['test', 'pred']])

elif mode == 'whole':

    sns.lineplot(data=train_compare[['test', 'pred']]), sns.lineplot(data=test_compare[['test', 'pred']])



display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top: 12.580)'.format(smape(y_test, y_pred), smape(y_train, y_self_pred)))

'''
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import PolynomialFeatures



items = np.sort(train['item'].unique())

stores = np.sort(train['store'].unique())

meta_sample = train.copy()



test_sample= test.copy()

test_sample['pred'] = 0.

#test_dummy['sales'] = 0



#meta_sample = meta_sample.append(test_dummy)

meta_sample['pred'] = 0.



# we need to train and validate model on train-val set and do predictions for part of the test set at the same loop iteration



for i in items:#[1]:

    for s in stores:#[1]: #stores:

        sub_sample = meta_sample[(meta_sample['item'] == i) & (meta_sample['store'] == s)]

        X_train, X_test, y_train, y_test = split(sub_sample)

        

        test_sub_sample = test_sample[(test_sample['item'] == i) & (test_sample['store'] == s)].drop('date', axis=1)

        

        

        ct = ColumnTransformer([#("onehot", OneHotEncoder(categories='auto'), ['store']),

                        ("poly", PolynomialFeatures(degree=2), ['year', 'month', 'day', 'weekday'])])

        poly_X_train = ct.fit_transform(X_train)

        poly_X_test = ct.transform(X_test)

        poly_real_test = ct.transform(test_sub_sample)

        

        linear_reg = Ridge(alpha=10)#alpha = 1000)

        linear_reg.fit(poly_X_train, y_train)

        

        X_train['pred'] = linear_reg.predict(poly_X_train)

        X_test['pred'] = linear_reg.predict(poly_X_test)

        test_sub_sample['pred'] = linear_reg.predict(poly_real_test)

        

        whole = X_train.append(X_test)

        

        meta_sample.loc[(meta_sample['item'] == i) & (meta_sample['store'] == s), 'pred'] = whole['pred'].values

        test_sample.loc[(test_sample['item'] == i) & (test_sample['store'] == s), 'pred'] = test_sub_sample['pred'].values



        display('Test score: {0:.3f}; Train score: {0:.3f}; (Kaggle top total: 12.580)'

        .format(smape(y_test, X_test['pred']), smape(y_train, X_train['pred'])))
output = test_sample.sort_index()['pred'].reset_index()



output = pd.DataFrame({'id': output.index,

                       'sales': output['pred']})

output.to_csv('submission.csv', index=False)
meta_sample[['sales', 'pred']].describe() #COOL
smape(meta_sample['sales'], meta_sample['pred']) #Good
whole
whole['pred'].values
len(X_train) + len(X_test)
test.set_index('id')
test
test_sample.sort_index()['pred'].reset_index()