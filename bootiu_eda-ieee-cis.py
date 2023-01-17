import os, time, gc, datetime

from datetime import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



pd.set_option('max_rows', 9999)

pd.set_option('max_columns', 9999)



import warnings

warnings.filterwarnings('ignore')



start = time.time()
# Memory Reducing

# https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee



def reduce_mem_usage(df):

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.select_dtypes(exclude=['object', 'category']).columns:

        if df[col].dtype != object:  # Exclude strings                       

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(99999,inplace=True)  # Null Data is Filled "99999"

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

    return df
# Data Loading

def load_data(frac=1.0):

    dtypes = {}



    cols = ['addr1', 'addr2', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'ProductCD',

            'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',

            'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',

            'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',

            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

            'DeviceType', 'DeviceInfo', 'P_emaildomain', 'R_emaildomain'

            ]



    dtypes.update({k: 'object' for k in cols})

    

    train_tr = pd.read_csv('../input/train_transaction.csv', dtype=dtypes)

    train_id = pd.read_csv('../input/train_identity.csv', dtype=dtypes)

    test_tr = pd.read_csv('../input/test_transaction.csv', dtype=dtypes)

    test_id = pd.read_csv('../input/test_identity.csv', dtype=dtypes)

    

    train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

    test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

    del train_tr, train_id, test_tr, test_id

    gc.collect()

    

    train = train.sample(frac=frac).reset_index(drop=True)

    

    return train, test
_train, _test = load_data()
_train.head()
_test.head()
print('Train shape: {}  Test shape: {}'.format(_train.shape, _test.shape))
null_df_train = pd.DataFrame(_train.isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(20, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:50])

plt.xticks(rotation=30)

plt.show()



null_df_test = pd.DataFrame(_test.isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_test['NULL'] = null_df_test['NULL'] / len(_train)

fig = plt.figure(figsize=(20, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_test[:50])

plt.xticks(rotation=30)

plt.show()



# Histogram of NA Ratio per Features

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 6), facecolor='w')

sns.distplot(null_df_train['NULL'], bins=20, kde=True, ax=axes[0])

sns.distplot(null_df_test['NULL'], bins=20, kde=True, ax=axes[1])

axes[0].set_title('NA Hist per features - Train Data')

axes[1].set_title('NA Hist per features - Test Data')

plt.show()
# # Delete Features which has >75% NA

# del_cols = null_df_train[null_df_train['NULL'] > 0.75]['index'].tolist()

# del_cols.extend(null_df_test[null_df_test['NULL'] > 0.75]['index'].tolist())

# del_cols = list(set(del_cols))



# # print(pd.Series(del_cols))



# _train.drop(del_cols, axis=1, inplace=True)

# _test.drop(del_cols, axis=1, inplace=True)
del null_df_train, null_df_test

gc.collect()
fig = plt.figure(figsize=(16, 4), facecolor='w')



_ = plt.hist(_train['TransactionDT'], bins=100), plt.hist(_test['TransactionDT'], bins=100)

plt.legend(['train','test'])

plt.show()
def prep_date(df):

    

    START_DATE = '2017-11-01'

    

    # Convert Date (TransactionDT(sec) is a difference from "2017/11/01")

    df['date'] = df['TransactionDT'].apply(lambda x: dt.strptime(START_DATE, '%Y-%m-%d') + datetime.timedelta(seconds=x))

    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    

    # Year, Month, Days, Weekdays, Hour, Minitue

    df['year'] = df['date'].apply(lambda x: x.year)

    df['month'] = df['date'].apply(lambda x: x.month)

    df['day'] = df['date'].apply(lambda x: x.day)

    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    df['hour'] = df['date'].apply(lambda x: x.hour)

    

    # Set Dtypes

    df['year'] = df['year'].astype('category')

    df['month'] = df['month'].astype('category')

    df['day'] = df['day'].astype('category')

    df['weekday'] = df['weekday'].astype('category')

    df['hour'] = df['hour'].astype('category')

    

    # Diff Now Time

    def diff_date(x):

        s = datetime.datetime(year=2019, month=8, day=25) - x

        return s.days

    df['Diff_from_Now'] = df['date'].apply(diff_date)

    

    return df





_train = prep_date(_train)

_test = prep_date(_test)
fig = plt.figure(figsize=(16, 4), facecolor='w')



_ = plt.hist(_train['date'], bins=100), plt.hist(_test['date'], bins=100)

plt.legend(['train','test'])

plt.show()
# Count isFraud = 0, 1 each year&month

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), facecolor='w')



train_0 = _train[_train['isFraud'] == 0].reset_index(drop=True)

train_1 = _train[_train['isFraud'] == 1].reset_index(drop=True)



a = train_0.groupby(['year', 'month'])['isFraud'].count().reset_index()

b = train_1.groupby(['year', 'month'])['isFraud'].count().reset_index()

a = pd.merge(a, b, on=['year', 'month'])

x = ['201711', '201712', '201801', '201802', '201803', '201804', '201805']

axes[0].bar(x, a['isFraud_x'])

axes[0].set_title('isFraud = 0')

axes[1].bar(x, a['isFraud_y'])

axes[1].set_title('isFraud = 1')

plt.show()



# Count isFraud = 0, 1 each weekday

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4), facecolor='w')



a = train_0.groupby('weekday')['isFraud'].count().reset_index()

b = train_1.groupby('weekday')['isFraud'].count().reset_index()

a = pd.merge(a, b, on=['weekday'])

x = ['Mon', 'Tue', 'Web', 'Thu', 'Fri', 'Sat', 'Sun']

axes[0].bar(x, a['isFraud_x'])

axes[0].set_title('isFraud = 0')

axes[1].bar(x, a['isFraud_y'])

axes[1].set_title('isFraud = 1')

plt.show()
del train_0, train_1, a, b

gc.collect()
fig = plt.figure(figsize=(14, 4), facecolor='w')

sns.countplot(x='ProductCD', data=_train, order=['W', 'C', 'R', 'H', 'S'])

plt.show()
fig = plt.figure(figsize=(14, 4), facecolor='w')

sns.countplot(x='ProductCD', data=_test, order=['W', 'C', 'R', 'H', 'S'])

plt.show()
# Imbalance Data

fig = plt.figure(figsize=(12, 4), facecolor='w')

sns.countplot(_train['isFraud'])

plt.show()
def plot_hist_matrix(train, num_cols=None, ncols=10, height=2):

    

    # Distribution target = 0 or 1

    if num_cols is None:

        num_cols = [c for c in train.select_dtypes(exclude=['object']).columns if c not in ['TransactionID', 'isFraud', 'TransactionDT', 'date', 

                                                                                            'year', 'month', 'day', 'weekday', 'hour', 'Diff_from_Now']]



    _train_0 = train[train['isFraud'] == 0].reset_index(drop=False)

    _train_1 = train[train['isFraud'] == 1].reset_index(drop=False)

    

    nrows = int(np.ceil(len(num_cols) / ncols))



    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, height * nrows), facecolor='w')



    for feature, ax in zip(num_cols, axes.ravel()):

        # Set Second Axis

        ax_2 = ax.twinx()

        sns.kdeplot(_train_0[feature], color='b', alpha=0.4, shade=True, ax=ax, legend=False)

        sns.kdeplot(_train_1[feature], color='r', alpha=0.4, shade=True, ax=ax_2, legend=False)

        ax.set_title(feature)

        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        ax_2.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    plt.tight_layout()

    plt.show()
plot_hist_matrix(_train)
c_cols = ['C{}'.format(i) for i in range(1, 15, 1)]

plot_hist_matrix(_train, num_cols=c_cols, ncols=4, height=3)
def plot_hist(train, colname):

    _train_0 = train[train['isFraud'] == 0].reset_index(drop=False)

    _train_1 = train[train['isFraud'] == 1].reset_index(drop=False)

    

    fig = plt.figure(figsize=(12, 4), facecolor='w')

    ax = sns.kdeplot(_train_0[colname], color='b', alpha=0.4, shade=True)

    ax_2 = ax.twinx()

    sns.kdeplot(_train_1[colname], color='r', alpha=0.4, shade=True, ax=ax_2)

    ax.set_title(colname)

    plt.show()
plot_hist(_train, 'C12')
def plot_hist_log(train, colname):

    _train_0 = train[train['isFraud'] == 0].reset_index(drop=False)

    _train_1 = train[train['isFraud'] == 1].reset_index(drop=False)  # isFraud=1: 20663

    

    fig = plt.figure(figsize=(12, 4), facecolor='w')

    ax = sns.kdeplot(np.log(_train_0[colname]), color='b', alpha=0.4, shade=True)

    ax_2 = ax.twinx()

    sns.kdeplot(np.log(_train_1[colname]), color='r', alpha=0.4, shade=True, ax=ax_2)

    ax.set_title(colname)

    plt.show()
plot_hist_log(_train, 'C12')

plt.show()
corr_table = _train[c_cols].corr()



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.heatmap(corr_table, vmin=0, vmax=1, center=0.5, square=True, cmap='Blues')

plt.show()
# PCA

_c_cols = c_cols + ['isFraud']



temp_df = _train.copy()

temp_df = temp_df[_c_cols]

temp_df = temp_df.fillna(99999)



pca = PCA(n_components=2)

a = pca.fit_transform(temp_df)



temp_df_2 = pd.DataFrame(a, columns=['PCA_1', 'PCA_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



temp_df = temp_df.sample(frac=0.05)



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='PCA_1', y='PCA_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
# t-SNE

temp_df = _train.copy()

temp_df = temp_df[_c_cols]

temp_df = temp_df.fillna(99999)



temp_df = temp_df.sample(frac=0.1)



tsne = TSNE(n_components=2, random_state=0)

a = tsne.fit_transform(temp_df[c_cols])



temp_df_2 = pd.DataFrame(a, columns=['TSNE_1', 'TSNE_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
def TimeSeriesPlot(col):

    fig = plt.figure(figsize=(12, 4), facecolor='w')

    

    # Cx values per transaction

    temp = _train.groupby('date_str')[col].sum() / _train.groupby('date_str')['TransactionID'].count()

    temp_2 = _test.groupby('date_str')[col].sum() / _test.groupby('date_str')['TransactionID'].count()

 

    _ = plt.plot(temp), plt.plot(temp_2)

    plt.xticks(np.arange(1, len(temp) + len(temp_2)+1, 30), rotation=30)

    plt.show()
TimeSeriesPlot('C3')
temp = _train.groupby('date_str')['C3'].sum() / _train.groupby('date_str')['TransactionID'].count()

temp.sort_values(ascending=False)[:5]
temp = _test.groupby('date_str')['C3'].sum() / _test.groupby('date_str')['TransactionID'].count()

temp.sort_values(ascending=False)[:5]
# Distrubition Card Info (card4, card6)

fig, axes = plt.subplots(1, 2, figsize=(12, 3), facecolor='w')

a = pd.DataFrame(_train['card4'].value_counts()).reset_index()

sns.barplot(x='card4', y='index', data=a, ax=axes[0])

a = pd.DataFrame(_train['card6'].value_counts()).reset_index()

sns.barplot(x='card6', y='index', data=a, ax=axes[1])

plt.tight_layout()

plt.show()
# Groupby card4, card6

for card in ['card4', 'card6']:

    for c in c_cols:

        temp_0 = _train[_train['isFraud'] == 0].groupby(card)[c].mean().reset_index()

        temp_1 = _train[_train['isFraud'] == 1].groupby(card)[c].mean().reset_index()

        temp = pd.merge(temp_0, temp_1, how='outer', on=card)

        temp = temp.fillna(0)



        fig, axes = plt.subplots(1, 2, figsize=(12, 3), facecolor='w')

        sns.barplot(x=f'{c}_x', y=card, data=temp, ax=axes[0])

        sns.barplot(x=f'{c}_y', y=card, data=temp, ax=axes[1])

        axes[0].set_title(f'{card}_{c} - isFraud=0')

        axes[1].set_title(f'{card}_{c} - isFraud=1')

        plt.tight_layout()

        plt.show()
del corr_table, temp, temp_df, temp_df_2

gc.collect()
fig = plt.figure(figsize=(12, 4), facecolor='w')



temp = _train.groupby('date_str')['TransactionAmt'].sum()

temp_2 = _test.groupby('date_str')['TransactionAmt'].sum()



_ = plt.plot(temp), plt.plot(temp_2)

plt.xticks(np.arange(1, len(temp) + len(temp_2)+1, 30), rotation=30)

plt.show()
# Display Top5 values (Train)

temp.sort_values(ascending=False)[:5]
# Display Top5 values (Test)

temp_2.sort_values(ascending=False)[:5]
fig = plt.figure(figsize=(18, 6), facecolor='w')

sns.boxplot(x='isFraud', y='TransactionAmt', data=_train)

plt.show()
# Outliner

# Drop TransactionAmt > 30000 

_train = _train.query('TransactionAmt < 30000')
# All Values is not Minus

len(_train[_train['TransactionAmt'] <= 0])
fig = plt.figure(figsize=(18, 6), facecolor='w')

sns.distplot(_train['TransactionAmt'])

plt.show()
fig = plt.figure(figsize=(18, 6), facecolor='w')

sns.distplot(np.log(_train['TransactionAmt']))

plt.show()
# Transaction Log Value

_train['log_TranAmt'] = np.log(_train['TransactionAmt'])

_test['log_TranAmt'] = np.log(_test['TransactionAmt'])
del temp, temp_2

gc.collect()
d_cols = ['D{}'.format(i) for i in range(1, 16, 1)]



plot_hist_matrix(_train, num_cols=d_cols, ncols=4, height=3)
col_list = d_cols + ['isFraud']

_train[col_list].head(10)
# Null Rate

null_df_train = pd.DataFrame(_train[d_cols].isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(24, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:100], order=d_cols)

plt.xticks(rotation=30)

plt.show()
_train[d_cols].describe()

# D4, D6, D11, D12, D14, D15 have Minus Values

# D9 values are between 0 and 0.95
_test[d_cols].describe()

# Test Data has no minus values
plot_hist(_train, 'D6')
plot_hist(_train, 'D11')
plot_hist(_train, 'D14')
corr_table = _train[d_cols].corr()



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.heatmap(corr_table, vmin=0, vmax=1, center=0.5, square=True, cmap='Blues')

plt.show()

# D9 is independence

# D1,D2 are high correlation
# PCA

_d_cols = d_cols + ['isFraud']



temp_df = _train.copy()

temp_df = temp_df[_d_cols]

temp_df = temp_df.fillna(99999)



pca = PCA(n_components=2)

a = pca.fit_transform(temp_df[d_cols])



temp_df_2 = pd.DataFrame(a, columns=['PCA_1', 'PCA_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='PCA_1', y='PCA_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
# t-SNE



temp_df = _train.copy()

temp_df = temp_df[_d_cols]

temp_df = temp_df.fillna(99999)



temp_df = temp_df.sample(frac=0.1)



tsne = TSNE(n_components=2, random_state=0)

a = tsne.fit_transform(temp_df[d_cols])



temp_df_2 = pd.DataFrame(a, columns=['TSNE_1', 'TSNE_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
col_list = d_cols + ['isFraud']



diff_dcols_df = _train[col_list]

_d_cols = d_cols.copy()

_d_cols.remove('D9')



for d_1 in _d_cols:

    for d_2 in _d_cols:

        if d_1 == d_2:

            continue

            

        diff_dcols_df[f'{d_1}-{d_2}'] = diff_dcols_df[d_1] - diff_dcols_df[d_2]

        
diff_dcols_df.head()
# PCA

# only use diff values

_d_cols = diff_dcols_df.columns[16:].tolist()



pca = PCA(n_components=2)

a = pca.fit_transform(diff_dcols_df[_d_cols].fillna(99999))



temp_df_2 = pd.DataFrame(a, columns=['PCA_1', 'PCA_2'])

temp_df_2['isFraud'] = diff_dcols_df['isFraud']



temp_df_2 = temp_df_2.sample(frac=0.05)



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='PCA_1', y='PCA_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
# t-SNE

# only use diff values

_d_cols = diff_dcols_df.columns[16:].tolist()

_temp_df = diff_dcols_df.sample(frac=0.1)



tsne = TSNE(n_components=2)

a = tsne.fit_transform(_temp_df[_d_cols].fillna(99999))



temp_df_2 = pd.DataFrame(a, columns=['TSNE_1', 'TSNE_2'])

temp_df_2['isFraud'] = _temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
y = diff_dcols_df['isFraud']

X = diff_dcols_df.drop('isFraud', axis=1)



X.fillna(-9999, inplace=True)



rf = RandomForestClassifier()

rf.fit(X, y)



importance_df = pd.DataFrame({

    'importance': rf.feature_importances_,

    'feature': X.columns

})



fig = plt.figure(figsize=(14, 40), facecolor='w')

sns.barplot(x='importance', y='feature', data=importance_df.sort_values(by='importance', ascending=False))

plt.show()



# D1, D8 are High Performance
del d_cols, _d_cols, col_list, diff_dcols_df, temp_df, temp_df_2, _temp_df, rf, importance_df

gc.collect()
m_cols = ['M{}'.format(i) for i in range(1, 10, 1)]

_train[m_cols].head()
# Null Rate

null_df_train = pd.DataFrame(_train[m_cols].isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(24, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:100], order=m_cols)

plt.xticks(rotation=30)

plt.show()
fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(16, 8), facecolor='w')



for c, ax in zip(m_cols, axes.ravel()):

    sns.countplot(x=c, ax=ax, data=_train[_train['isFraud'] == 0])

    ax.set_title(c)

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(16, 8), facecolor='w')



for c, ax in zip(m_cols, axes.ravel()):

    sns.countplot(x=c, ax=ax, data=_train[_train['isFraud'] == 1])

    ax.set_title(c)

plt.tight_layout()

plt.show()
del m_cols, null_df_train

gc.collect()
v_cols = ['V{}'.format(i) for i in range(1, 340, 1)]



_train[v_cols].head()
# Null Rate

null_df_train = pd.DataFrame(_train[v_cols].isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(24, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:100], order=v_cols)

plt.xticks(rotation=30)

plt.show()
corr_table = _train[v_cols].corr()



fig = plt.figure(figsize=(24, 18), facecolor='w')

sns.heatmap(corr_table, vmin=0, vmax=1, center=0.5, square=True, cmap='Blues')

plt.show()
# PCA

_v_cols = v_cols + ['isFraud']



temp_df = _train.copy()

temp_df = temp_df[_v_cols]

temp_df = temp_df.fillna(99999)



pca = PCA(n_components=2)

a = pca.fit_transform(temp_df[v_cols])



temp_df_2 = pd.DataFrame(a, columns=['PCA_1', 'PCA_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='PCA_1', y='PCA_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
del v_cols, _v_cols, null_df_train, corr_table, temp_df, temp_df_2

gc.collect()
id_cols_1 = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11']



plot_hist_matrix(_train, num_cols=id_cols_1, ncols=4, height=3)
_train[id_cols_1].describe()
_test[id_cols_1].describe()
# Null Rate

null_df_train = pd.DataFrame(_train[id_cols_1].isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(24, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:100], order=id_cols_1)

plt.xticks(rotation=30)

plt.show()
# PCA

_id_cols_1 = id_cols_1 + ['isFraud']



temp_df = _train.copy()

temp_df = temp_df[_id_cols_1]

temp_df = temp_df.fillna(99999)



pca = PCA(n_components=2)

a = pca.fit_transform(temp_df[id_cols_1])



temp_df_2 = pd.DataFrame(a, columns=['PCA_1', 'PCA_2'])

temp_df_2['isFraud'] = temp_df['isFraud']



fig = plt.figure(figsize=(12, 6), facecolor='w')

sns.scatterplot(x='PCA_1', y='PCA_2', hue='isFraud', data=temp_df_2, alpha=0.4)

plt.legend()

plt.show()
del id_cols_1, temp_df, temp_df_2, null_df_train

gc.collect()
id_cols_2 = ['id_{}'.format(i) for i in range(12, 39, 1)]



ncols = 5

height = 3

nrows = int(np.ceil(len(id_cols_2) / ncols))



train_0 = _train[_train['isFraud'] == 0]

train_1 = _train[_train['isFraud'] == 1]



fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, height * nrows), facecolor='w')

for feature, ax in zip(id_cols_2, axes.ravel()):

    sns.countplot(x=feature, hue='isFraud', ax=ax, data=pd.concat([train_0.sample(20000), train_1.sample(20000)], axis=0))

    ax.set_title(feature)

    

plt.tight_layout()

plt.show()
# Null Rate

null_df_train = pd.DataFrame(_train[id_cols_2].isnull().sum(), columns=['NULL']).sort_values(by='NULL', ascending=False).reset_index()

null_df_train['NULL'] = null_df_train['NULL'] / len(_train)

fig = plt.figure(figsize=(24, 4), facecolor='w')

sns.barplot(x='index', y='NULL', data=null_df_train[:100], order=id_cols_2)

plt.xticks(rotation=30)

plt.show()