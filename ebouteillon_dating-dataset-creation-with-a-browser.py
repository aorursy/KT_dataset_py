from datetime import datetime, timedelta

import gc

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

df = pd.concat([train, test], sort=True)

del train_transaction, test_transaction, train_identity, test_identity, train, test

gc.collect();
x = df.groupby('id_31').id_31.count().sort_values(ascending=False).head(30)

plt.figure(figsize=(16, 10))

plt.title('Most frequent browsers in dataset')

sns.barplot(x=x, y=x.index);
x = [x for x in df.id_31.unique().astype(str) if 'chrome' in x]

x = df[df.id_31.isin(x)].groupby('id_31').id_31.count().sort_values(ascending=False)

plt.figure(figsize=(16, 16))

plt.title('All chrome version present in dataset')

sns.barplot(x=x, y=x.index);
df['day'] = df.TransactionDT // (3600*24)
def chrome_frequency(platform):

    plt.figure(figsize=(16, 10))

    for i in range(39, 72):

        browser = f'chrome {i}.0{platform}'

        if browser in df.id_31.unique().tolist():

            z = df[df.id_31==browser].groupby('day').day.count().rename('y').reset_index()

            plt.title(f'Chrome{platform}: Count of transactions per version and per day')

            sns.lineplot(x='day', y='y', data=z, label=browser)

    plt.show()
chrome_frequency('')
chrome_frequency(' for android')
chrome_frequency(' for ios')
# version, desktop, android, ios

release = pd.DataFrame([

    [39, '2014-11-18', '2014-11-12', '2014-11-24'],

    [40, '2015-01-21', '2015-01-21', '2015-01-20'],

    [41, '2015-03-03', '2015-03-11', '2015-03-16'],

    [42, '2015-04-14', '2015-04-15', '2015-04-16'],

    [43, '2015-05-19', '2015-05-27', '2015-06-01'],

    [44, '2015-07-21', '2015-07-29', '2015-07-22'],

    [45, '2015-09-01', '2015-09-01', '2015-09-02'],

    [46, '2015-10-13', '2015-10-14', '2015-10-22'],

    [47, '2015-12-01', '2015-12-02', '2015-12-02'],

    [48, '2016-01-20', '2016-01-27', '2016-01-27'],

    [49, '2016-03-02', '2016-03-09', '2016-03-09'],

    [50, '2016-04-13', '2016-04-26', '2016-04-20'],

    [51, '2016-05-25', '2016-06-01', '2016-06-01'],

    [52, '2016-07-20', '2016-07-27', '2016-07-27'],

    [53, '2016-08-31', '2016-09-07', '2016-09-07'],

    [54, '2016-10-12', '2016-10-19', '2016-10-19'],

    [55, '2016-12-01', '2016-12-06', '2016-12-05'],

    [56, '2017-01-25', '2017-02-01', '2017-02-01'],

    [57, '2017-03-09', '2017-03-16', '2017-03-14'],

    [58, '2017-04-19', '2017-04-20', '2017-04-25'],

    [59, '2017-06-05', '2017-06-06', '2017-06-06'],

    [60, '2017-07-25', '2017-07-31', '2017-07-25'],

    [61, '2017-09-05', '2017-09-05', '2017-09-05'],

    [62, '2017-10-17', '2017-10-19', '2017-10-18'],

    [63, '2017-12-06', '2017-12-05', '2017-12-05'],

    [64, '2018-01-24', '2018-01-23', '2018-01-24'],

    [65, '2018-03-06', '2018-03-06', '2018-03-06'],

    [66, '2018-04-17', '2018-04-17', '2018-04-17'],

    [67, '2018-05-29', '2018-05-31', '2018-05-29'],

    [68, '2018-07-24', '2018-07-24', '2018-07-24'],

    [69, '2018-09-04', '2018-09-04', '2018-09-04'],

    [70, '2018-10-16', '2018-10-17', '2018-10-16'],

    [71, '2018-12-04', '2018-12-04', '2018-12-04'],

    [72, '2019-01-29', '2019-01-29', '2019-01-29'],],

    columns=['version', 'desktop', 'android', 'ios'])
result = pd.DataFrame(columns=['browser', 'first_day'])

for col, platform in enumerate(['', ' for android', ' for ios']):

    for i in range(63, 72):

        browser = f'chrome {i}.0{platform}'

        if browser in df.id_31.unique().tolist():

            shift = df[df.id_31 == browser].TransactionDT.min() // (3600*24)

            release_date = release.iloc[i-39][col+1]

            result = result.append({

                'browser': browser,

                'first_day': datetime.strptime(release_date, '%Y-%m-%d') - timedelta(days=shift),

                'version': i

            }, ignore_index=True)



result
result.first_day.quantile(.5)