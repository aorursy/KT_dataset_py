import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import gc

from itertools import cycle, islice

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
del train_transaction, train_identity

del test_transaction, test_identity

gc.collect()
def downcast_df_float_columns(df):

    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)

        

    if len(list_of_columns)>=1:

        max_string_length = max([len(col) for col in list_of_columns])

        print("downcasting float for:", list_of_columns, "\n")

        

        for col in list_of_columns:

            df[col] = pd.to_numeric(df[col], downcast="float")

    else:

        print("no columns to downcast")

    gc.collect()

    print("done")
def downcast_df_int_columns(df):

    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)

        

    if len(list_of_columns)>=1:

        max_string_length = max([len(col) for col in list_of_columns])

        print("downcasting integers for:", list_of_columns, "\n")

        

        for col in list_of_columns:

            df[col] = pd.to_numeric(df[col], downcast="integer")

    else:

        print("no columns to downcast")

    gc.collect()

    print("done")
train.head()
train.info()
train.describe()
plt.bar(['train', 'test'], [len(train), len(test)], width=0.2, color='b')

plt.title("Number of train set and test set instances ")

plt.show()
plt.bar(['train', 'test'], [train.isnull().any().sum(), test.isnull().any().sum()], width=0.2, color='g')

plt.title("Number of column with null values")

plt.show()
train_missing_values = train.isnull().sum().sort_values(ascending=False) / len(train)

test_missing_values = test.isnull().sum().sort_values(ascending=False) / len(test)



fig, axes = plt.subplots(2, 1, figsize=(12, 8))

sns.barplot(list(train_missing_values.keys()[:10]), train_missing_values[:10], ax=axes[0])

sns.barplot(list(test_missing_values.keys()[:10]), test_missing_values[:10], ax=axes[1])

plt.show()
def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.2f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)



plt.figure(figsize=(5, 4))

ax = sns.barplot(["fraud", "not fraud"],

            [len(train[train.isFraud == 1])/len(train),

             len(train[train.isFraud == 0])/len(train)])

show_values_on_bars(ax)

plt.show()
train["hour"] = np.floor(train["TransactionDT"] / 3600) % 24

test["hour"] = np.floor(train["TransactionDT"] / 3600) % 24
plt.plot(train.groupby('hour').mean()['isFraud'], color='r')

ax = plt.gca()

ax2 = ax.twinx()

_ = ax2.hist(train['hour'], alpha=0.3, bins=24)

ax.set_xlabel('Encoded hour')

ax.set_ylabel('Fraction of fraudulent transactions')



ax2.set_ylabel('Number of transactions')

plt.show()
train["DeviceType"].value_counts(dropna=False).plot.bar()

plt.show()
plt.figure(figsize=(8, 8))

sns.barplot(train["DeviceInfo"].value_counts(dropna=False)[:15], 

            train["DeviceInfo"].value_counts(dropna=False).keys()[:15])

plt.show()
my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(train.P_emaildomain.value_counts())))

train.P_emaildomain.value_counts().plot.bar(figsize=(20, 10), color=my_colors)

plt.show()
plt.figure(figsize=(6, 6))

plt.pie([np.sum(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values),

                                 len(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values) - 

                                 np.sum(train[(train['P_emaildomain'] == 'protonmail.com')].isFraud.values)],

        labels=['isFraud', 'notFraud'], autopct='%1.1f%%')

plt.show()
train['is_proton_mail'] = (train['P_emaildomain'] == 'protonmail.com') | (train['R_emaildomain']  == 'protonmail.com')

test['is_proton_mail'] = (test['P_emaildomain'] == 'protonmail.com') | (test['R_emaildomain']  == 'protonmail.com')
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',

          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',

          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 

          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',

          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',

          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',

          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',

          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',

          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')



print('done')
a = np.zeros(train.shape[0])

train["lastest_browser"] = a

a = np.zeros(test.shape[0])

test["lastest_browser"] = a

def setbrowser(df):

    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1

    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1

    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1

    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1

    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1

    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1

    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1

    return df

train=setbrowser(train)

test=setbrowser(test)
plt.figure(figsize=(6, 6))

plt.pie([np.sum(train[(train['lastest_browser'] == True)].isFraud.values),

                                 len(train[(train['lastest_browser'] == True)].isFraud.values) - 

                                 np.sum(train[(train['lastest_browser'] == True)].isFraud.values)],

        labels=['isFraud', 'notFraud'], autopct='%1.1f%%', colors=['y', 'g'])

plt.show()
train_missing_values = [str(x) for x in train_missing_values[train_missing_values > 0.80].keys()]

test_missing_values = [str(x) for x in test_missing_values[test_missing_values > 0.80].keys()]



dropped_columns = train_missing_values + test_missing_values
dropped_columns = dropped_columns + [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

dropped_columns = dropped_columns + [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

dropped_columns.remove('isFraud')



train.drop(dropped_columns, axis=1, inplace=True)

test.drop(dropped_columns, axis=1, inplace=True)



len(dropped_columns)
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')
train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)

test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)



train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)

test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)



train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)

test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)



train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)

test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)



train['TransactionAmt'] = np.log1p(train['TransactionAmt'])

test['TransactionAmt'] = np.log1p(test['TransactionAmt'])    
for feature in ['id_36']:

    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        

for feature in ['id_01', 'id_31', 'id_35', 'id_36']:

    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
for col in ['card1']: 

    valid_card = pd.concat([train[[col]], test[[col]]])

    valid_card = valid_card[col].value_counts()

    valid_card = valid_card[valid_card>2]

    valid_card = list(valid_card.index)



    train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)

    test[col]  = np.where(test[col].isin(train[col]), test[col], np.nan)



    train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)

    test[col]  = np.where(test[col].isin(valid_card), test[col], np.nan)
numerical_columns = list(test.select_dtypes(exclude=['object']).columns)



train[numerical_columns] = train[numerical_columns].fillna(train[numerical_columns].median())

test[numerical_columns] = test[numerical_columns].fillna(train[numerical_columns].median())

print("filling numerical columns null values done")
categorical_columns = list(filter(lambda x: x not in numerical_columns, list(test.columns)))

categorical_columns[:5]
train[categorical_columns] = train[categorical_columns].fillna(train[categorical_columns].mode())

test[categorical_columns] = test[categorical_columns].fillna(train[categorical_columns].mode())

print("filling numerical columns null values done")
from sklearn.preprocessing import LabelEncoder



for col in categorical_columns:

    le = LabelEncoder()

    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

    train[col] = le.transform(list(train[col].astype(str).values))

    test[col] = le.transform(list(test[col].astype(str).values))
labels = train["isFraud"]

train.drop(["isFraud"], axis=1, inplace=True)
X_train, y_train = train, labels

del train, labels

gc.collect()
lgb_submission=sample_submission.copy()

lgb_submission['isFraud'] = 0
n_fold = 5

folds = KFold(n_fold)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

    print(fold_n)

    

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)

    

    lgbclf = lgb.LGBMClassifier(

            num_leaves= 512,

            n_estimators=512,

            max_depth=9,

            learning_rate=0.064,

            subsample=0.85,

            colsample_bytree=0.85,

            boosting_type= "gbdt",

            reg_alpha=0.3,

            reg_lamdba=0.243

    )

    

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]

    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    lgbclf.fit(X_train_,y_train_)

    

    del X_train_,y_train_

    print('finish train')

    pred=lgbclf.predict_proba(test)[:,1]

    val=lgbclf.predict_proba(X_valid)[:,1]

    print('finish pred')

    del lgbclf, X_valid

    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))

    del val,y_valid

    lgb_submission['isFraud'] = lgb_submission['isFraud']+ pred/n_fold

    del pred

    gc.collect()
lgb_submission.insert(0, "TransactionID", np.arange(3663549, 3663549 + 506691))

lgb_submission.to_csv('prediction.csv', index=False)