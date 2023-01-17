# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re, string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression,ElasticNet,Ridge

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_table('../input/train.tsv')

test = pd.read_table('../input/test.tsv')

train_index = train.index

test_index = test.index



# Any results you write to the current directory are saved as output.
def make_columns(df):

    if 'price' in df:

        df['logprice'] = np.log1p(df['price'])

    df.loc[df['category_name'].isnull(),'category_name'] = 'NA/NA/NA'

    df['cat0'] = df['category_name'].map(lambda x: x.split('/')[0])

    df['cat1'] = df['category_name'].map(lambda x: x.split('/')[1])

    df['cat2'] = df['category_name'].map(lambda x: x.split('/')[2])

    pattern = re.compile('[^a-zA-Z0-9 ]+', re.UNICODE)

    df['cleaned_name'] = df['name'].map(lambda x: ' '.join(filter(None,pattern.sub(' ', x.lower()).strip().split(' '))))
make_columns(train)

make_columns(test)
def bayes_average(data_ins,data_oos1,data_oos2,names,alpha):

    sums = data_ins.groupby(names)['logprice'].sum()

    counts = data_ins.groupby(names)['logprice'].count()

    mean = data_ins['logprice'].mean()

    m = (sums + alpha*mean)/(counts + alpha)

    mean1 = data_ins.loc[data_ins.groupby(names)['logprice'].transform('count') == 1, 'logprice'].mean()

    dd1 = data_oos1.join(m, how='left',on=names,rsuffix='_pred')

    dd1.loc[dd1['logprice_pred'].isnull(),'logprice_pred'] = mean1

    dd2 = data_oos2.join(m, how='left',on=names,rsuffix='_pred')

    dd2.loc[dd2['logprice'].isnull(),'logprice'] = mean1

    return dd1['logprice_pred'],dd2['logprice']
train1, train2 = np.split(train.sample(frac=1), [int(.7*len(train))])

names = ['brand_name','cleaned_name','category_name','cat1','cat2',['category_name','shipping','item_condition_id']]

for name in names:

    alpha = 1

    y_pred_validate, y_pred_test = bayes_average(train1,train2,test,name,alpha)

    train2['_'.join(name)+'_pred'] = y_pred_validate

    test['_'.join(name)+'_pred'] = y_pred_test

lm = Ridge(alpha=1)

X_val = train2[['_'.join(x)+'_pred' for x in names]]

y_val = train2['logprice']

lm.fit(X_val,y_val)

X_test = test[['_'.join(x)+'_pred' for x in names]]

y_pred = lm.predict(X_test)

y_pred_exp = np.expm1(y_pred)

submission = pd.DataFrame(y_pred_exp, columns=['price'])

submission.to_csv('./submission.csv', index_label='test_id')