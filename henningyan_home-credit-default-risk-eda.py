import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
# import the dataset
import os
listdir = os.listdir("../input")
# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

print(listdir)
df_train = pd.read_csv("../input/application_train.csv")
df_test = pd.read_csv("../input/application_test.csv")
bureau_balance = pd.read_csv("../input/bureau_balance.csv")
pos_cash = pd.read_csv("../input/POS_CASH_balance.csv")
previous_application = pd.read_csv("../input/previous_application.csv")
installments_payments = pd.read_csv("../input/installments_payments.csv")
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv")
bureau = pd.read_csv("../input/bureau.csv")
previous_application.head()
pos_cash.head()
installments_payments.head()
credit_card_balance.head()
bureau.head()
bureau.isnull().sum()
def agg_status(x):
    status = list(x.values)
    a = np.unique(status, return_counts=True)
    return (a[0][a[1].argmax()])

serie_group_bureau = bureau_balance.groupby(['SK_ID_BUREAU'])['STATUS'].apply(agg_status)
bu_balance_group = pd.DataFrame(columns = ['SK_ID_BUREAU','STATUS_CASH', 'NUMBER_LOAN']) 
bu_balance_group['SK_ID_BUREAU'] = serie_group_bureau.index.values
bu_balance_group['STATUS_CASH'] = serie_group_bureau.values
bu_balance_group['NUMBER_LOAN'] = bureau_balance.groupby(['SK_ID_BUREAU']).count().values
bu_balance_group.head()
bureau_full = pd.merge(bureau, bu_balance_group, on= 'SK_ID_BUREAU')
bureau_full.head(10)
bureau_full.isnull().sum()
bureau_full_f = bureau_full.fillna(0)
bureau_full_f[bureau_full_f['SK_ID_CURR']==100002]
bureau_once = pd.DataFrame(columns=['SK_ID_CURR', 'CREDIT_ACTIVE', 'CREDIT_CLOSED', 'DAYS_CREDIT_SUM', 'CREDIT_DAY_OVERDUE_SUM',
                                   'DAYS_CREDIT_ENDDATE_SUM', 'DAYS_CREDIT_ENDDATE_FACT_SUM', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG',
                                    'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE_MAX',
                                   'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY', 'STATUS', 'NUMBER_LOAN'])

bureau_once['SK_ID_CURR'] = bureau_full_f.groupby(['SK_ID_CURR']).count().index.values
bureau_once['CREDIT_ACTIVE'] = bureau_full_f.groupby(['SK_ID_CURR'])['CREDIT_ACTIVE'].apply(lambda x: (x=='Active').sum()).values
bureau_once['CREDIT_CLOSED'] = bureau_full_f.groupby(['SK_ID_CURR'])['CREDIT_ACTIVE'].apply(lambda x: (x=='Closed').sum()).values
bureau_once['CREDIT_CURRENCY'] = bureau_full_f.groupby(['SK_ID_CURR'])['CREDIT_CURRENCY'].apply(agg_status).values
bureau_once['DAYS_CREDIT_SUM'] = bureau_full_f.groupby(['SK_ID_CURR'])['DAYS_CREDIT'].sum().values
bureau_once['CREDIT_DAY_OVERDUE_SUM'] = bureau_full_f.groupby(['SK_ID_CURR'])['CREDIT_DAY_OVERDUE'].sum().values
bureau_once['DAYS_CREDIT_ENDDATE_SUM'] = bureau_full_f.groupby(['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].sum().values
bureau_once['DAYS_CREDIT_ENDDATE_FACT_SUM'] = bureau_full_f.groupby(['SK_ID_CURR'])['DAYS_ENDDATE_FACT'].sum().values
bureau_once['AMT_CREDIT_MAX_OVERDUE'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_CREDIT_MAX_OVERDUE'].sum().values
bureau_once['CNT_CREDIT_PROLONG'] = bureau_full_f.groupby(['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].sum().values
bureau_once['AMT_CREDIT_SUM'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().values
bureau_once['AMT_CREDIT_SUM_DEBT'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().values
bureau_once['AMT_CREDIT_SUM_LIMIT'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_LIMIT'].sum().values
bureau_once['AMT_CREDIT_SUM_OVERDUE'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().values
bureau_once['CREDIT_TYPE_MAX'] = bureau_full_f.groupby(['SK_ID_CURR'])['CREDIT_TYPE'].apply(agg_status).values
bureau_once['DAYS_CREDIT_UPDATE'] = bureau_full_f.groupby(['SK_ID_CURR'])['DAYS_CREDIT_UPDATE'].sum().values
bureau_once['AMT_ANNUITY'] = bureau_full_f.groupby(['SK_ID_CURR'])['AMT_ANNUITY'].sum().values
bureau_once['STATUS'] = bureau_full_f.groupby(['SK_ID_CURR'])['STATUS_CASH'].apply(agg_status).values
bureau_once['NUMBER_LOAN'] = bureau_full_f.groupby(['SK_ID_CURR'])['NUMBER_LOAN'].sum().values

bureau_once.head()
bureau_once[bureau_once['SK_ID_CURR']==100002]
#df_train_full.head(20)
#df_train.head()
df_train_full = pd.merge(df_train, bureau_once, on='SK_ID_CURR')
df_test_full = pd.merge(df_test, bureau_once, on='SK_ID_CURR')
# check the basic train data
df_train.head()
# make a statistic analysis in the numeric features
df_train.describe()
df_train.info()
# check for null values and where they are
def search_missing_data(df):
    null_features = df.isnull().sum()
    null_features = [null_features.index[x] for x in range(len(df.isnull().sum())) if null_features[x] > 0]
    type_features = [df[x].dtype for x in null_features]
    return null_features, type_features
features_miss_train = pd.DataFrame()
features_miss_test = pd.DataFrame()

null_features, type_features = search_missing_data(df_train)
features_miss_train['feature'] = null_features
features_miss_train['dtype'] = type_features

null_features, type_features = search_missing_data(df_test)
features_miss_test['feature'] = null_features
features_miss_test['dtype'] = type_features
print('Features missing in the train set', features_miss_train.shape[0], '\nFeatures missing in the test set', features_miss_test.shape[0])
for x in features_miss_train['feature'].values:
    if x not in features_miss_test['feature'].values:
        print(x)
features_miss_test['feature'].values
print(df_train.shape[0],df_train.dropna().shape[0])
def preprocess_data(df, test=0):
    df1 = df
    df_dict = {0:features_miss_train['feature'].values,
              1: features_miss_test['feature'].values}
    for feature in df_dict[test]:
        if df1[feature].dtype == 'object':
            df1[feature].fillna('None', inplace=True)
        if df1[feature].dtype == 'float64':
            df1[feature].fillna(0.0, inplace=True)
    df1['CODE_GENDER'].replace('XNA', 'F',inplace=True)
    df1['NAME_INCOME_TYPE'].replace('Maternity leave', 'Student',inplace=True)
    df1['NAME_FAMILY_STATUS'].replace('Unknown', 'Single / not married', inplace=True)
    
    return df1
df_prep_train = preprocess_data(df_train)
df_prep_test = preprocess_data(df_test, test=1)
features_to_plot = list(df_prep_train.columns.values)

features_part_one = features_to_plot[2:32]
features_part_two = features_to_plot[32:62]
features_part_three = features_to_plot[62:92]
features_part_four = features_to_plot[92:122]

def plot_charts(features):
    ncols = 5
    nrows = round(len(features)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
    n_feature = 0
    for row in range(nrows):
        for col in range(ncols):
            if n_feature < len(features):
                name = features[n_feature]
                if df_prep_train[name].dtype == 'object' or df_prep_train[name].dtype == 'int64':
                    g = sns.countplot(hue='TARGET', y = name, data = df_prep_train, ax = axes[row][col])
                    g.set_ylabel(name)
                    g.set_title('TARGET')
                else:
                    g = sns.boxplot(x='TARGET', y = name, data = df_prep_train, ax = axes[row][col])
                    g.set_ylabel(name)
                    g.set_title('TARGET')
            n_feature += 1
    # [plt.setp(ax.get_xticklabels(), rotation=90) for ax in axes.flat]
    plt.tight_layout()
    plt.show()
plot_charts(features_part_one)
plot_charts(features_part_two)
plot_charts(features_part_three)
plot_charts(features_part_four)
df_dummy_train = pd.get_dummies(df_prep_train.drop(['TARGET', 'SK_ID_CURR'], axis=1), drop_first=True)
df_dummy_train['TARGET'] = df_prep_train['TARGET']
df_dummy_train['SK_ID_CURR'] = df_prep_train['SK_ID_CURR']
df_dummy_test = pd.get_dummies(df_prep_test, drop_first=True)

list_train = list(df_dummy_train.columns.values)
list_test = list(df_dummy_test.columns.values)

for f in list_train:
    if f not in list_test:
        print(f)

print(df_dummy_train.shape)
print(df_dummy_test.shape)

clt = lgb.LGBMClassifier()
X_train = df_dummy_train.drop(['TARGET', 'SK_ID_CURR'], axis=1).values
y_train = df_dummy_train['TARGET'].values
X_test = df_dummy_test.drop(['SK_ID_CURR'], axis=1).values
id_test = df_dummy_test['SK_ID_CURR'].values
clt.fit(X_train, y_train)
y_test = clt.predict(X_test)
submission = pd.DataFrame()
submission['SK_ID_CURR'] = id_test
submission['TARGET'] = y_test
submission.to_csv('submission2.csv', index=False)