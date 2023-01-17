# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt

import seaborn as sns

get_ipython().magic('matplotlib inline')



pd.options.display.max_rows = 100
df = pd.read_csv('/kaggle/input/is-this-a-good-customer/clients.csv')
df.shape
df.head()
numeric_columns = ['credit_amount', 'credit_term', 'age', 'income']

categorical_columns = ['month', 'sex', 'education', 'product_type', 'having_children_flg', 'region', 'family_status', 'phone_operator', 'is_client']
df.describe()
def scale_range(inp, min, max):

    inp += (-np.min(inp))

    inp /= np.max(inp) / (max-min)

    inp += min

    return inp



def plot_rate(df, column, target, num_buckets=5):

    data = pd.DataFrame(df[column], columns=[column,])

    data['target'] = target

    tmp_ = data.dropna().groupby(column).agg({'target': ['mean', sum, 'count']})

    tmp_['index'] = range(tmp_.shape[0])

    tmp = pd.DataFrame({'index': range(tmp_.shape[0]), 'value': tmp_.index, 'mean': tmp_['target']['mean'], 

                        'sum': tmp_['target']['sum'], 'count': tmp_['target']['count']})

    breakpoints = np.arange(0, num_buckets+1) / (num_buckets) * 100

    breakpoints = scale_range(breakpoints, np.min(tmp_['index'].values), np.max(tmp_['index'].values))

    percents = np.histogram(tmp_['index'], breakpoints)[0]

    

    l = []

    for bucket in range(num_buckets):

        l = l + [bucket] * int(percents[bucket])

        

    ll = []

    if (tmp_.shape[0] > len(l)) | (tmp_.shape == 2):

        ll = []

        for j in l:

            ll.append(j)

        if tmp.shape[0] > len(ll):

            tmp['bucket'] = ll + [max(ll)]

        else:

            tmp['bucket'] = ll

    else:

        tmp['bucket'] = l

    

    result = tmp.groupby('bucket').agg({'sum': 'sum', 'count': 'sum', 'value': [min, max]})

    result['avg_target'] = result['sum']['sum'] / result['count']['sum']

    

    plt.figure()

    plt.suptitle(column+' avg_target', size=16)

    sns.barplot(y=result['value']['min'].astype('int64'), x=result['avg_target'], orient='h')

    

    plt.figure()

    plt.suptitle(column+' count', size=16)

    sns.barplot(y=result['value']['min'].astype('int64'), x=result['count']['sum'], orient='h')



sns.distplot(df['credit_amount']);
plot_rate(df, 'credit_amount', df['bad_client_target'], num_buckets=20)
sns.distplot(df['credit_term']);
plot_rate(df, 'credit_term', df['bad_client_target'], num_buckets=20)
sns.distplot(df['age']);
plot_rate(df, 'age', df['bad_client_target'], num_buckets=20)
sns.distplot(df['income']);
plot_rate(df, 'income', df['bad_client_target'], num_buckets=20)
df[categorical_columns].describe(include='all')
df['month'].value_counts().plot(kind='bar');
sns.catplot(x="month", y="bad_client_target", kind="bar", data=df);
df[['month', 'bad_client_target']].groupby('month').agg('mean').sort_values('bad_client_target', ascending=False)
df['sex'].value_counts().plot(kind='bar');
sns.catplot(x="sex", y="bad_client_target", kind="bar", data=df);
df[['sex', 'bad_client_target']].groupby('sex').agg('mean').sort_values('bad_client_target', ascending=False)
df['sex'] = df['sex'].apply(lambda x: 1 if x=='male' else 0)
df['education'].value_counts().plot(kind='bar');
cp = sns.catplot(x="education", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=60);
df[['education', 'bad_client_target']].groupby('education').agg('mean').sort_values('bad_client_target', ascending=False)
df['product_type'].value_counts().plot(kind='bar');
cp = sns.catplot(x="product_type", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['product_type', 'bad_client_target']].groupby('product_type').agg('mean').sort_values('bad_client_target', ascending=False)
df['having_children_flg'].value_counts().plot(kind='bar');
cp = sns.catplot(x="having_children_flg", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['having_children_flg', 'bad_client_target']].groupby('having_children_flg').agg('mean').sort_values('bad_client_target', ascending=False)
df['region'].value_counts().plot(kind='bar');
cp = sns.catplot(x="region", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['region', 'bad_client_target']].groupby('region').agg('mean').sort_values('bad_client_target', ascending=False)
df['family_status'].value_counts().plot(kind='bar');
cp = sns.catplot(x="family_status", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['family_status', 'bad_client_target']].groupby('family_status').agg('mean').sort_values('bad_client_target', ascending=False)
df['phone_operator'].value_counts().plot(kind='bar');
cp = sns.catplot(x="phone_operator", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['phone_operator', 'bad_client_target']].groupby('phone_operator').agg('mean').sort_values('bad_client_target', ascending=False)
df['is_client'].value_counts().plot(kind='bar');
cp = sns.catplot(x="is_client", y="bad_client_target", kind="bar", data=df)

cp.set_xticklabels(rotation=90);
df[['is_client', 'bad_client_target']].groupby('is_client').agg('mean').sort_values('bad_client_target', ascending=False)
df['bad_client_target'].value_counts().plot(kind='bar');