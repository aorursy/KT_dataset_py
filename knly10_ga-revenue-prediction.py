import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%%time
# credit to https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
def load_df(csv_path='../input/ga-customer-revenue-prediction/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    json = __import__('json')
    json_normalize = pd.io.json.json_normalize
    df = pd.read_csv(
        csv_path,
        converters={ column: json.loads for column in JSON_COLUMNS },
        dtype={ 'fullVisitorId': 'str', 'visitId': 'str' },
        nrows=nrows
    )
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f'{column}.{subcolumn}' for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f'Loaded {csv_path}. Shape: {df.shape}')
    return df

# df_train = load_df()
# df_test = load_df('../input/ga-customer-revenue-prediction/test.csv')
# %%time
# df_train.to_csv('train-flattened.csv', index=False)
# df_test.to_csv('test-flattened.csv', index=False)
%%time
# these have been saved earlier
df_train = pd.read_csv('../input/ga-store-customer-revenue/train-flattened.csv', dtype={ 'fullVisitorId': str, 'visitId': str, 'trafficSource.campaignCode': str },)
df_test = pd.read_csv('../input/ga-store-customer-revenue/test-flattened.csv', dtype={ 'fullVisitorId': str, 'visitId': str },)
df_train['totals.transactionRevenue'] = df_train['totals.transactionRevenue'].astype(float)
df_train['totals.transactionRevenue'].notnull().sum()
np.sum(df_train.groupby('fullVisitorId')['totals.transactionRevenue'].sum() > 0)
df_train['totals.transactionRevenue'].sum() / 1000000
(df_train['totals.transactionRevenue'][df_train['totals.transactionRevenue'] > 0] / 1000000).agg(['min', 'max', 'mean', 'median'])
columns = set(df_train.columns).intersection((set(df_test.columns))) # columns present in both train and test
print('Predictors that can be removed:', end=' ')
for i in df_train.columns[df_train.nunique(dropna=False) == 1].tolist() + df_test.columns[df_test.nunique(dropna=False) == 1].tolist():
    if (i in columns):
        print(i, end=', ')
        columns.remove(i)
def count(var, p=True):
    tmp = pd.concat([
        df_train[var].value_counts(dropna=False),
        df_train[df_train['totals.transactionRevenue'] > 0][var].value_counts(dropna=False),
        df_test[var].value_counts(dropna=False),
    ], keys=['train', 'train with revenue', 'test'], axis=1, sort=False)
    tmp['train %'] = np.round(tmp['train'] / tmp['train'].sum() * 100, 1)
    tmp['train with revenue %'] = np.round(tmp['train with revenue'] / tmp['train with revenue'].sum() * 100, 1)
    tmp['test %'] = np.round(tmp['test'] / tmp['test'].sum() * 100, 1)
    if p: print(tmp.shape)
    return(tmp)
count('channelGrouping')
count('device.browser')
count('device.deviceCategory')
count('device.operatingSystem')
count('geoNetwork.continent')
count('geoNetwork.subContinent')
count('geoNetwork.country')
count('geoNetwork.region')
count('geoNetwork.city')
count('geoNetwork.metro')
count('geoNetwork.networkDomain')
count('trafficSource.adContent')
count('trafficSource.adwordsClickInfo.adNetworkType')
count('trafficSource.adwordsClickInfo.gclId')
count('trafficSource.adwordsClickInfo.isVideoAd')
count('trafficSource.adwordsClickInfo.page')
count('trafficSource.adwordsClickInfo.slot')
count('trafficSource.campaign')
count('trafficSource.isTrueDirect')
count('trafficSource.keyword')
count('trafficSource.medium')
count('trafficSource.referralPath')
count('trafficSource.source')
count('totals.bounces')
count('totals.hits')
count('totals.pageviews')
count('totals.newVisits')
for i in ['sessionId', 'visitId', 'fullVisitorId']:
    print(i, 'Nunique:', df_train[i].nunique(), '. Null:', df_train[i].isnull().sum())
    print('Common between train and test set:', len(set(df_train[i]).intersection(set(df_test[i]))))

# fullVisitorId present in both train and test set
tmp = list(set(df_train['fullVisitorId']).intersection(set(df_test['fullVisitorId'])))
# the revenue of these fullVisitorId
tmp = df_train[df_train['fullVisitorId'].isin(tmp)].groupby('fullVisitorId')['totals.transactionRevenue'].sum()
print('Number of fullVisitorId who spent money:', len(tmp[tmp > 0]), '. Total amount spent:', tmp.sum() / 1e6)

# fullVisitorId only present in train
tmp = list(set(df_train['fullVisitorId']).difference(set(df_test['fullVisitorId'])))
# the revenue of these fullVisitorId
tmp = df_train[df_train['fullVisitorId'].isin(tmp)].groupby('fullVisitorId')['totals.transactionRevenue'].sum()
print('Number of fullVisitorId only in the train set:', len(tmp), '. Number who spent money:', len(tmp[tmp > 0]), '. Total amount spent:', tmp.sum() / 1e6)
tmp = df_train.groupby('fullVisitorId')['totals.transactionRevenue'].agg(['size', 'count', 'sum']).groupby('size').agg(['size', 'sum'])[[['count', 'size'], ['count', 'sum'], ['sum', 'sum']]]
tmp.columns = ['size', 'count', 'revenue']
tmp['revenue'] = tmp['revenue'] / 1e6
tmp['size %'] = np.round(tmp['size'] / tmp['size'].sum() * 100, 2)
tmp['count / size %'] = np.round(tmp['count'] / tmp['size'] * 100, 2)
tmp['revenue %'] = np.round(tmp['revenue'] / tmp['revenue'].sum() * 100, 2)
tmp
plt.plot(tmp.index, tmp['count / size %']);
plt.xlabel('Frequency of fullVisitorId')
plt.ylabel('Percentage of number of transaction with revenue\nover number of fullVisitorId')
plt.xlim(0, 50)
plt.ylim(0, 500)
tmp_date = pd.to_datetime(df_train['date'], format='%Y%m%d')
tmp_visitStartTime = pd.to_datetime(df_train['visitStartTime'], unit='s')
(tmp_date.dt.date - tmp_visitStartTime.dt.date).value_counts() # this shows most date and visitStartTime are the same

# new columns
df_train['visitStartTime.month'] = tmp_visitStartTime.dt.month
df_train['visitStartTime.week'] = tmp_visitStartTime.dt.week
df_train['visitStartTime.day'] = tmp_visitStartTime.dt.day
df_train['visitStartTime.weekday'] = tmp_visitStartTime.dt.weekday
df_train['visitStartTime.hour'] = tmp_visitStartTime.dt.hour

# do the same for test_df
tmp_date = pd.to_datetime(df_test['date'], format='%Y%m%d')
tmp_visitStartTime = pd.to_datetime(df_test['visitStartTime'], unit='s')
(tmp_date.dt.date - tmp_visitStartTime.dt.date).value_counts() # this shows most date and visitStartTime are the same

df_test['visitStartTime.month'] = tmp_visitStartTime.dt.month
df_test['visitStartTime.week'] = tmp_visitStartTime.dt.week
df_test['visitStartTime.day'] = tmp_visitStartTime.dt.day
df_test['visitStartTime.weekday'] = tmp_visitStartTime.dt.weekday
df_test['visitStartTime.hour'] = tmp_visitStartTime.dt.hour
def make_button(var, title = '', max_row = 10, df = df_train, target_var = 'totals.transactionRevenue'):
    # create data for the 'updatemenus' used by plotly
    # agg data from var ~ target_vaar into size, count, mean, median
    # return dict()
    tmp = df[[var, target_var]].fillna(value={ var: -1 }).groupby(var)[target_var].agg(['size', 'sum', 'mean', 'median']) # use fillna for var, as groupby(var) doesn't work with na
    tmp = tmp.sort_values('size', ascending=False)[:max_row][::-1] # by defaul, take only the top 10 rows ordered by size
    tmp = {
        'x': [tmp['size'].values, tmp['sum'].values , tmp['mean'].values, tmp['median'].values],
        'y': [[str(i) for i in tmp.index.tolist()]] * 4, # str(i) to convert all to string, because some of the indexes are True, False
    }
    title = title or var
    return dict(args=[tmp, { 'title': title }], label=title, method='update') 

# plotting
## data
tmp = make_button('device.deviceCategory', 'Device Category')
x = tmp['args'][0]['x']
y = tmp['args'][0]['y'][0]

## trace
traces = [None] * 4
traces[0] = (go.Bar(x=x[0], y=y, orientation='h'))
traces[1] = (go.Bar(x=x[1], y=y, orientation='h'))
traces[2] = (go.Bar(x=x[3], y=y, orientation='h', name='Median')) # median goes first, the resulting bar graph will place median at the bottom
traces[3] = (go.Bar(x=x[2], y=y, orientation='h', name='Mean'))

## fig, subplot
fig = __import__('plotly').tools.make_subplots(1, 3, subplot_titles=['Number of record', 'Total revenue', 'Mean & Median'])
for i in range(3): fig.append_trace(traces[i], 1, i + 1)
fig.append_trace(traces[-1], 1, 3)

## fig, layout
fig.layout.title = tmp['args'][1]['title']
fig.layout.showlegend = False
fig.layout.updatemenus = list([
    dict(
        buttons=[make_button(i) for i in [
            'device.deviceCategory', 'device.operatingSystem', 'device.browser', 'device.isMobile',
            'geoNetwork.continent', 'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city', 'geoNetwork.networkDomain',
            'trafficSource.medium', 'trafficSource.campaign', 'trafficSource.isTrueDirect', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
        ]] + [make_button(i, max_row=31) for i in [
            'totals.bounces', 'totals.newVisits', 'totals.hits', 'totals.pageviews',
            'visitStartTime.month', 'visitStartTime.week', 'visitStartTime.day', 'visitStartTime.weekday', 'visitStartTime.hour'
        ]],
        direction = 'down',
        showactive = True,
        x = 0,
        xanchor = 'left',
        y = 1.25,
        yanchor = 'top' 
    ),
])

## now plot
py.iplot(fig)

# clean up for memory
%xdel make_button
%xdel tmp
%xdel traces
%xdel fig
# Convert target `totals.transactionRevenue` with log1p
df_train['totals.transactionRevenue'].fillna(0, inplace=True)
y = np.log1p(df_train['totals.transactionRevenue'])
plt.hist(y[y > 0], bins=50);
# fillna
print('Number of records which are null in the train and test set')
pd.concat([df_train[list(columns)].isnull().sum(0), df_test[list(columns)].isnull().sum(0)], keys=['train', 'test'], axis=1).sort_index()

# fillna with 0
for i in ['totals.bounces', 'totals.newVisits', 'totals.pageviews', 'trafficSource.adwordsClickInfo.page']:
    df_train[i].fillna(0, inplace=True)
    df_test[i].fillna(0, inplace=True)

# fillna with the word "NONE"
for i in ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.keyword', 'trafficSource.referralPath']:
    df_train[i].fillna('NONE', inplace=True)
    df_test[i].fillna('NONE', inplace=True)

# change True, False to 1, 0
df_train['trafficSource.adwordsClickInfo.isVideoAd'].replace({ np.nan: 1, False: 0 }, inplace=True); df_test['trafficSource.adwordsClickInfo.isVideoAd'].replace({ np.nan: 1, False: 0 }, inplace=True)
df_train['trafficSource.isTrueDirect'].replace({ np.nan: 0, True: 1 }, inplace=True); df_test['trafficSource.isTrueDirect'].replace({ np.nan: 0, True: 1 }, inplace=True)
df_train['device.isMobile'].replace({ True: 1, False: 0 }, inplace=True); df_test['device.isMobile'].replace({ True: 1, False: 0 }, inplace=True)

print('Number of records which are null in the train and test set after fillna')
pd.concat([df_train[list(columns)].isnull().sum(0), df_test[list(columns)].isnull().sum(0)], keys=['train', 'test'], axis=1).sum()
%%time
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

def encode_oh(var, threshold = 0.0005, ohe = OneHotEncoder(handle_unknown='ignore')):
    levels = df_train[var].value_counts()
    levels = levels[levels / len(df_train) > 0.0005].index.values.reshape(-1, 1) # need to use the threshold, otherwise the dimensions explode and eat up all the memory
    ohe.fit(levels)
    train = ohe.transform(df_train[var].values.reshape(-1, 1))
    test = ohe.transform(df_test[var].values.reshape(-1, 1))
    feature_names = [var + '__' + i[3:] for i in ohe.get_feature_names()]
    return train, test, feature_names

# LabelBinarizer
# def encode_lb(var, threshold = 0.0001, array = None):
#     lb = LabelEncoder()
#     # var, the var to be encoded    
#     # thresold, default at 0.0001 or 0.01%. Keep columns where at least 0.01% of rows are not 0
#     # array, prodive an array of categories to be fitted
#     lb.fit(array or df_train[var])
#     train = lb.transform(df_train[var])
#     test = lb.transform(df_test[var])
#     columns = [var + '__' + str(i) for i in lb.classes_]
#     columns_to_keep = train.sum(0) / train.sum() > thresold
#     train = train[:, columns_to_keep]
#     test = test[:, columns_to_keep]
#     columns = np.array(columns)[columns_to_keep]    
#     return train, test, columns
    
# predictors

# categorical variables + numeric variables
tmp = [encode_oh(i) for i in [
    'channelGrouping',
    'device.browser', 'device.deviceCategory', 'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
    'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source',
]] + [(sparse.csr_matrix(df_train[i].values.reshape(-1, 1)), sparse.csr_matrix(df_test[i].values.reshape(-1, 1)), [i]) for i in [
    'device.isMobile', 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews', 'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.page', 'trafficSource.isTrueDirect', 'visitNumber', 'visitStartTime.month', 'visitStartTime.week', 'visitStartTime.day', 'visitStartTime.weekday', 'visitStartTime.hour'
]]

X = sparse.hstack([i[0] for i in tmp]).tocsr()
X_test = sparse.hstack([i[1] for i in tmp])
columns = np.concatenate([i[2] for i in tmp])

# # free up memory
%xdel sparse
%xdel encode_oh
%xdel tmp
%%time
from sklearn.preprocessing import LabelEncoder
tmp_cats = [
    'channelGrouping',
    'device.browser', 'device.deviceCategory', 'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent',
    'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign', 'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath', 'trafficSource.source',
    'sessionId', 'visitId', 'fullVisitorId', # fullVisitorId could be useful
]
tmp_nums = ['device.isMobile', 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews', 'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.page', 'trafficSource.isTrueDirect', 'visitNumber', 'visitStartTime', 'visitStartTime.month', 'visitStartTime.week', 'visitStartTime.day', 'visitStartTime.weekday', 'visitStartTime.hour']

# Xl: X using labelEncoder
Xl = df_train[tmp_cats + tmp_nums].copy()
Xl_test = df_test[tmp_cats + tmp_nums].copy()

l = LabelEncoder()
print('Processing:', end=' ')
for i in tmp_cats:
    print(i, end=', ')
    # because labelencoder cannot encode unseen labels
    Xl[i] = l.fit_transform(Xl[i]) + 1 # all existing labels will be +1
    in_index = Xl_test[i].isin(l.classes_) # select rows in test_df where labels have been seen 
    Xl_test.loc[in_index, i] = l.transform(Xl_test[i][in_index]) + 1
    Xl_test.loc[~in_index, i] = 0 # all unseen label will be coded 0
#     l.fit(np.concatenate([Xl[i].values.astype(str), Xl_test[i].values.astype(str)]))
#     Xl[i] = l.transform(Xl[i].values.astype(str))
#     Xl_test[i] = l.transform(Xl_test[i].values.astype(str))

# free up memory
%xdel l
%xdel tmp_cats
%xdel tmp_nums
%%time
import pickle

for filename, obj in [
    ('df_train', df_train), ('df_test', df_test),
    ('X', X), ('X_test', X_test),
    ('Xl', Xl), ('Xl_test', Xl_test),
]:
    with open(filename + '.pickle', 'wb') as handle: pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)