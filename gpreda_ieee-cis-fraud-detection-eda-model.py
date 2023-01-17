import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import re

from sklearn import metrics, preprocessing

from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA

import lightgbm as lgb



pd.set_option('display.max_columns', 400)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_identity_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'train_identity.csv'))

train_transaction_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'train_transaction.csv'))

test_identity_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'test_identity.csv'))

test_transaction_df = pd.read_csv(os.path.join('../input/ieee-fraud-detection/', 'test_transaction.csv'))
print(f"train identity: {train_identity_df.shape}")

print(f"train transaction: {train_transaction_df.shape}")

print(f"test identity: {test_identity_df.shape}")

print(f"test transaction: {test_transaction_df.shape}")
train_identity_df.head()
train_transaction_df.head()
test_identity_df.head()
test_transaction_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(train_identity_df)
missing_data(test_identity_df)
missing_data(train_transaction_df)
missing_data(test_transaction_df)
df = pd.DataFrame()

for column in list(train_identity_df.columns.values):

    field_type = train_identity_df[column].dtype

    df=df.append(pd.DataFrame({'column':column,'train':train_identity_df[column].nunique(), 'test':test_identity_df[column].nunique(),\

                               'type':field_type},index=[0]))

df['delta'] = df.train - df.test

df['flag'] = (df['delta'] < 0).astype(int)

test_dom_categories = df.loc[(df.flag == 1) & (df.type == 'object'), 'column']

df = df.transpose()



print('Unique column values in identity datasets')

df
print(f"Columns of type `object` and with more categories in test than in train: {list(test_dom_categories)}")
df = pd.DataFrame()

for column in list(test_transaction_df.columns.values):

    field_type = test_transaction_df[column].dtype

    try:

        df=df.append(pd.DataFrame({'column':column,'train':train_transaction_df[column].nunique(), \

            'test':test_transaction_df[column].nunique(), 'type':field_type},index=[0]))

    except:

        "Error trying to add target from test"

df['delta'] = df.train - df.test

df['flag'] = (df['delta'] < 0).astype(int)

test_dom_categories = df.loc[(df.flag == 1) & (df.type == 'object'), 'column']

df = df.transpose()



print('Unique column values in transaction datasets')

df
print(f"Columns of type `object` and with more categories in test than in train: {list(test_dom_categories)}")
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()   
plot_count('id_30', 'train: id_30', df=train_identity_df, size=4)
plot_count('id_30', 'test: id_30', df=test_identity_df, size=4)
plot_count('id_31', 'train: id_31', df=train_identity_df, size=4)
plot_count('id_31', 'test: id_31', df=test_identity_df, size=4)
plot_count('id_33', 'train: id_33', df=train_identity_df, size=4)
plot_count('id_33', 'test: id_33', df=test_identity_df, size=4)
plot_count('DeviceInfo', 'train: DeviceInfo', df=train_identity_df, size=4)
plot_count('DeviceInfo', 'test: DeviceInfo', df=test_identity_df, size=4)
plot_count('P_emaildomain', 'train: P_emaildomain', df=train_transaction_df, size=4)
plot_count('P_emaildomain', 'test: P_emaildomain', df=test_transaction_df, size=4)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
identity_cols = list(train_identity_df.columns.values)

transaction_cols = list(train_transaction_df.drop('isFraud', axis=1).columns.values)
X_train = pd.merge(train_transaction_df[transaction_cols + ['isFraud']], train_identity_df[identity_cols], how='left')

X_train = reduce_mem_usage(X_train)

X_test = pd.merge(test_transaction_df[transaction_cols], train_identity_df[identity_cols], how='left')

X_test = reduce_mem_usage(X_test)
X_train_id = X_train.pop('TransactionID')

X_test_id = X_test.pop('TransactionID')

del train_identity_df,train_transaction_df, test_identity_df, test_transaction_df
all_data = X_train.append(X_test, sort=False).reset_index(drop=True)
vcols = [f'V{i}' for i in range(1,340)]



sc = preprocessing.MinMaxScaler()



pca = PCA(n_components=2) #0.99

vcol_pca = pca.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))



all_data['_vcol_pca0'] = vcol_pca[:,0]

all_data['_vcol_pca1'] = vcol_pca[:,1]

all_data['_vcol_nulls'] = all_data[vcols].isnull().sum(axis=1)



all_data.drop(vcols, axis=1, inplace=True)
all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)

all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)

all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)

all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)

all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)

all_data['_card_all__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)

all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)

all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))

all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))

cols = ['ProductCD','card1','card2','card5','card6','P_emaildomain','_card_all__addr1']



for f in cols:

    all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')

    all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')

    all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']



for f in cols:

    vc = all_data[f].value_counts(dropna=False)

    all_data[f'_count_{f}'] = all_data[f].map(vc)

    

cat_cols = [f'id_{i}' for i in range(12,39)]

for i in cat_cols:

    if i in all_data.columns:

        all_data[i] = all_data[i].astype(str)

        all_data[i].fillna('unknown', inplace=True)



enc_cols = []

for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():

    if t == object:

        enc_cols.append(i)

        all_data[i] = pd.factorize(all_data[i])[0]  
X_train = all_data[all_data['isFraud'].notnull()]

X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)

Y_train = X_train.pop('isFraud')

del all_data
params={'learning_rate': 0.0097,

        'objective': 'binary',

        'metric': 'auc',

        'num_threads': -1,

        'num_leaves': 256,

        'verbose': 1,

        'random_state': 314,

        'bagging_fraction': 1,

        'feature_fraction': 0.82

       }
oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])



clf = lgb.LGBMClassifier(**params, n_estimators=4000)

clf.fit(X_train, Y_train)

oof_preds = clf.predict_proba(X_train, num_iteration=clf.best_iteration_)[:,1]

sub_preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]
submission = pd.DataFrame()

submission['TransactionID'] = X_test_id

submission['isFraud'] = sub_preds

submission.to_csv('submission.csv', index=False)