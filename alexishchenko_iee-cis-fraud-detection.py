# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import datetime,random

import lightgbm as lgb

import scipy.stats as stats

import gc

import warnings

from time import time

from sklearn import metrics

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
def dataframe_cat_feature_summary(df):

    summary = pd.DataFrame(df.dtypes,columns = ["dtypes"])

    summary = summary.reset_index()

    summary["Feature"] = summary["index"]

    summary = summary[["Feature","dtypes"]]

    summary["Missed"] = df.isna().sum().values

    df_cat_describe = df.describe()

    summary["Count"] = df_cat_describe.iloc[0,:].values

    summary["% Missed"] = (summary["Missed"] / df.shape[0]) * 100

    summary["% Missed"] = round(summary["% Missed"], 2)

    summary["Unique"] = df_cat_describe.iloc[1,:].values

    summary["Top"] = df_cat_describe.iloc[2,:].values

    summary["Freq"] = df_cat_describe.iloc[3,:].values

    

    return summary



def dataframe_num_feature_summary(df):

    summary = pd.DataFrame(df.dtypes,columns = ["dtypes"])

    summary = summary.reset_index()

    summary["Feature"] = summary["index"]

    summary = summary[["Feature","dtypes"]]

    summary["Missed"] = df.isna().sum().values

    df_num_describe = df.describe()

    summary["Count"] = df_num_describe.iloc[0,:].values

    summary["% Missed"] = (summary["Missed"] / df.shape[0]) * 100

    summary["% Missed"] = round(summary["% Missed"], 2)

    summary["Mean"] = df_num_describe.iloc[1,:].values

    summary["Std"] = df_num_describe.iloc[2,:].values

    summary["Min"] = df_num_describe.iloc[3,:].values

    summary["Max"] = df_num_describe.iloc[7,:].values

    summary["25 %"] = df_num_describe.iloc[4,:].values

    summary["75 %"] = df_num_describe.iloc[6,:].values

    

    return summary



def impute(df, columns_to_impute, imputer):

    imputed_df = pd.DataFrame(imputer.fit_transform(df[columns_to_impute]))

    imputed_df.columns = columns_to_impute # From Kaggele Cources to work with missing values

    df[columns_to_impute] = imputed_df



def ploting_cnt_amt_m(df, column):

    fraud_explanation = pd.crosstab(df[column], df["isFraud"], normalize = "index") * 100

    fraud_explanation = fraud_explanation.reset_index()

    fraud_explanation.rename(columns={0:"NoFraud", 1:"Fraud"}, inplace = True)

    

    plt.figure(figsize=(20,10))

    plt.subplot(211)

    plt.suptitle(f"{column} Dist" , fontsize=24)

    

    order_of_values_on_charts = df[column].unique()

    

    chart_feature = sns.countplot(x = column,  data = df, order = order_of_values_on_charts)

    chart_feature.set_xticklabels(chart_feature.get_xticklabels(),rotation=45,horizontalalignment = "right")

    

    chart_fraud = chart_feature.twinx()

    chart_fraud = sns.pointplot(x = column, y = "Fraud", data = fraud_explanation, order = order_of_values_on_charts, color = "red")

    

    plt.show()

    

def ploting_numeric_features(df, column):

    plt.figure(figsize=(20,10))

    plt.subplot(211)

    percent_of_nan = np.round(100*np.sum(df[column].isna())/df.shape[0],2)

    plt.suptitle(column +' has '+str(df[column].nunique())+' '+'values'+' and '+str(percent_of_nan)+'% nan' , fontsize=24)

    h = plt.hist(df[column].dropna(),bins=50) 

    plt.show()
SEED = 42



def seed_env(seed=SEED):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df



pd.set_option('display.max_columns', 380)

pd.set_option('display.max_rows', 500)

seed_env()
features_to_delete = list()
train_transactions_df = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv'))

train_identity_df = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv'))

train_df = pd.merge(train_transactions_df,train_identity_df,on = "TransactionID", how = "left")

target = train_df["isFraud"].copy()



del train_transactions_df, train_identity_df

gc.collect()



train_df.drop(["TransactionID","isFraud"], axis=1, inplace = True)



train_df.head()
def list_of_cat_features(base_name,feature_range):

    return [base_name + str(i) for i in feature_range]
categorical_features = ["ProductCD","P_emaildomain","R_emaildomain","DeviceType","DeviceInfo"]



categorical_features.extend(list_of_cat_features("card",range(1,7)))

categorical_features.extend(list_of_cat_features("addr",range(1,3)))

categorical_features.extend(list_of_cat_features("M",range(1,10)))

categorical_features.extend(list_of_cat_features("id_",range(12,39)))
numeric_features = list(set(train_df.columns.tolist()) - set(categorical_features))
params = {'num_leaves': 2**8,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': SEED

         }
non_important_features = set()



n_splits = 2

folds = KFold(n_splits = n_splits)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    train, y_train_df = train_df.iloc[trn_idx], target.iloc[trn_idx]

    valid, y_valid_df = train_df.iloc[val_idx], target.iloc[val_idx]

    

    trn_data = lgb.Dataset(train, label=y_train_df)

    val_data = lgb.Dataset(valid, label=y_valid_df)

    

    clf = lgb.train(params,trn_data,500,valid_sets = [trn_data, val_data],verbose_eval=200,early_stopping_rounds=200)



    pred = clf.predict(valid)

    #print("AUC = ", metrics.roc_auc_score(y_valid_df, pred))

    

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),train.columns), reverse=True), columns=["Value","Feature"])

    #print(feature_imp)

                                                                                                           

    non_important_features.update(set(feature_imp[feature_imp["Value"] < 10]["Feature"]))                                                                                                           

                                                                                                           

    visual_chunk_n = int(feature_imp.shape[0] / 50)

    tail_n = feature_imp.shape[0]  -  int(visual_chunk_n*50) - 1

                                                                                                           

    for i in range(0,visual_chunk_n):

        plt.figure(figsize=(16,16))

        plt.subplot(211)

        importance_bar = sns.barplot(data=feature_imp.iloc[i*50 : (i+1)*50 - 1], x='Value', y='Feature')

        plt.show()

                                                                                                           

    plt.figure(figsize=(16,16))

    plt.subplot(211)

    importance_bar_final = sns.barplot(data=feature_imp.tail(tail_n + 1), x='Value', y='Feature')

    plt.show()

                                                                                                           

    del trn_data,val_data,train, valid

    gc.collect()
nans_df = train_df.isna()

nans_groups={}



for col in train_df.columns:

    cur_group = nans_df[col].sum()

    try:

        nans_groups[cur_group].append(col)

    except:

        nans_groups[cur_group]=[col]

del nans_df; x=gc.collect()



for k,v in nans_groups.items():

    print('NAN count =',k)

    print(v)
analyze_groups = [

['D1', 'V281', 'V282', 'V283', 'V288', 'V289', 'V296', 'V300', 'V301', 'V313', 'V314', 'V315'],

['D8', 'D9', 'id_09', 'id_10'],

['D11', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11'],

['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34'],

['V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52'],

['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74'],

['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94'],

['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137'],

['V138', 'V139', 'V140', 'V141', 'V142', 'V146', 'V147', 'V148', 'V149', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V161', 'V162', 'V163'],

['V143', 'V144', 'V145', 'V150', 'V151', 'V152', 'V159', 'V160', 'V164', 'V165', 'V166'],

['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V186', 'V187', 'V190', 'V191', 'V192', 'V193', 'V196', 'V199', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216'],

['V169', 'V170', 'V171', 'V174', 'V175', 'V180', 'V184', 'V185', 'V188', 'V189', 'V194', 'V195', 'V197', 'V198', 'V200', 'V201', 'V208', 'V209', 'V210'],

['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237', 'V240', 'V241', 'V242', 'V243', 'V244', 'V246', 'V247', 'V248', 'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278'],

['V220', 'V221', 'V222', 'V227', 'V234', 'V238', 'V239', 'V245', 'V250', 'V251', 'V255', 'V256', 'V259', 'V270', 'V271', 'V272'],

['V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321'],

['V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339'],

['id_01', 'id_12'],

['id_22', 'id_23', 'id_27']

]
def display_corr_map(df,cols):

    plt.figure(figsize=(15,15))

    sns.heatmap(df[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)

    plt.title(cols[0]+' - '+cols[-1],fontsize=14)

    plt.show()
for group in analyze_groups:

    display_corr_map(train_df,['TransactionDT'] + group)
corr_features = set()



for group in analyze_groups:

    group_corr_matrix = train_df[['TransactionDT'] + group].corr().abs().unstack().sort_values(kind="quicksort")

    group_corr_matrix = group_corr_matrix[group_corr_matrix != 1][group_corr_matrix > 0.75].drop_duplicates()

    corr_features.update([x[1] for x in group_corr_matrix.index])
features_to_delete = list(corr_features.union(non_important_features))
print(len(features_to_delete))

categorical_features = list(set(categorical_features) - set(features_to_delete))

numeric_features = list(set(numeric_features) - set(features_to_delete))
del corr_features, non_important_features, analyze_groups,nans_groups, folds, train_df

gc.collect()
train_transactions_df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train_identity_df = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_df = pd.merge(train_transactions_df,train_identity_df,on = "TransactionID", how = "left")

y = train_df["isFraud"].copy()



test_transactions_df = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

test_identity_df = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_df = pd.merge(test_transactions_df,test_identity_df,on = "TransactionID", how = "left")



del train_transactions_df, train_identity_df,test_transactions_df,test_identity_df

gc.collect()
train_df.drop(features_to_delete + ["TransactionID"], axis=1, inplace = True)

test_df.drop(features_to_delete + ["TransactionID"], axis=1, inplace = True)



gc.collect()

train_df.head()
train_df.shape
train_df[categorical_features] = train_df[categorical_features].astype("object")

df_cat_summary = dataframe_cat_feature_summary(train_df[categorical_features])

df_cat_summary
train_df_cat_no_missing = train_df[categorical_features + ["isFraud"]].fillna("Missed")

for feature in categorical_features:

    ploting_cnt_amt_m(train_df_cat_no_missing, feature)
cat_columns_to_drop = df_cat_summary[df_cat_summary["% Missed"] > 90]["Feature"].values

print(cat_columns_to_drop)

train_df.drop(cat_columns_to_drop, axis=1, inplace = True)
imputer_most_frequent = SimpleImputer(strategy='most_frequent')

categorical_features = list(set(categorical_features) - set(cat_columns_to_drop))

impute(train_df, categorical_features, imputer_most_frequent)
df_num_summary = dataframe_num_feature_summary(train_df[numeric_features])

df_num_summary
for feature in numeric_features:

    ploting_numeric_features(train_df, feature)
num_columns_to_drop = df_num_summary[df_num_summary["% Missed"] > 90]["Feature"].values

print(num_columns_to_drop)

train_df.drop(num_columns_to_drop, axis=1, inplace = True)

numeric_features = list(set(numeric_features) - set(num_columns_to_drop))
imputer_mean = SimpleImputer(strategy='mean')

impute(train_df, numeric_features, imputer_mean)
#Formal check that there are no missing values for data frame

train_df.isnull().sum().sum()
numeric_features_to_log = list()

for feature in numeric_features:

    skeweness = stats.skew(train_df[feature])

    print("f = {0}, s = {1}".format(feature,skeweness))

    if skeweness > 13:

        numeric_features_to_log.append(feature)
def encode_cat_features(train, test, cat_features):

    for col in cat_features:

        label_encoder = LabelEncoder()

        label_encoder.fit(list(train[col].values) + list(test[col].values))

        train[col] = label_encoder.transform(list(train[col].values))

        test[col] = label_encoder.transform(list(test[col].values))

        

def standarize_num_features(df, num_features):

    scaler = MinMaxScaler()

    df[num_features] = scaler.fit_transform(df[num_features])

    

def decrease_skeweness(df, features):

    print(features)

    df[features] = np.log(df[features])
test_df.drop(cat_columns_to_drop, axis=1, inplace = True)

test_df.drop(num_columns_to_drop, axis=1, inplace = True)



imputer_test_mean = SimpleImputer(strategy='mean')

numeric_features_test = list(set(numeric_features) - set(['isFraud']))

impute(test_df, numeric_features_test, imputer_test_mean)
impute(test_df, categorical_features, imputer_most_frequent)

encode_cat_features(train_df,test_df,categorical_features)
test_df.isnull().sum().sum()
def add_features(df):

    START_DATE = '2017-12-01'

    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    

    df['year'] = df['TransactionDT'].dt.year

    df['month'] = df['TransactionDT'].dt.month

    df['dow'] = df['TransactionDT'].dt.dayofweek

    df['hour'] = df['TransactionDT'].dt.hour

    df['day'] = df['TransactionDT'].dt.day

    

    

    

    cards_cols= ['card1', 'card2', 'card3', 'card5']

    for card in cards_cols: 

        if '1' in card: 

            df['card_id']= df[card].map(str)

        else : 

            df['card_id']+= df[card].map(str)

    

    df['card_id'] = df['card_id'].astype(np.int64)
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

add_features(train_df)

add_features(test_df)
def add_aggregations(df):

    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')

    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')

    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')

    df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')

    df['Trans_min_mean'] = df['TransactionAmt'] - df['TransactionAmt'].mean()

    df['Trans_min_std'] = df['Trans_min_mean'] / df['TransactionAmt'].std()

    df['D15_to_mean_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('mean')

    df['D15_to_mean_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('mean')

    df['D15_to_std_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('std')

    df['D15_to_std_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('std')

    df['D15_to_mean_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('mean')

    df['D15_to_mean_addr2'] = df['D15'] / df.groupby(['addr2'])['D15'].transform('mean')

    df['D15_to_std_addr1'] = df['D15'] / df.groupby(['addr1'])['D15'].transform('std')

    df['D15_to_std_addr2'] = df['D15'] / df.groupby(['addr2'])['D15'].transform('std')

    df['id_02_to_mean_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('mean')

    df['id_02_to_mean_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('mean')

    df['id_02_to_std_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('std')

    df['id_02_to_std_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('std')
add_aggregations(train_df)

add_aggregations(test_df)
def remove_one_value_features(df,df_test):

    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]

    one_value_cols_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]

    df.drop(list(set(one_value_cols+ one_value_cols_test)), axis=1, inplace=True)

    df_test.drop(list(set(one_value_cols+ one_value_cols_test)), axis=1, inplace=True)

    

def remove_big_top_values_features(df,df_test,columns):

    big_top_value_cols = [col for col in columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    big_top_value_cols_test = [col for col in columns if df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    df.drop(list(set(big_top_value_cols+ big_top_value_cols_test)), axis=1, inplace=True)

    df_test.drop(list(set(big_top_value_cols+ big_top_value_cols_test)), axis=1, inplace=True)
remove_one_value_features(train_df,test_df)

remove_big_top_values_features(train_df,test_df,numeric_features_test)
X = train_df.drop(["isFraud", "TransactionDT"], axis=1)

del train_df

gc.collect()
X_test = test_df.sort_values("TransactionDT").drop(["TransactionDT"], axis=1)

del test_df

gc.collect()
X.head()
X_test.head()
folds = TimeSeriesSplit(n_splits=5)
params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': SEED

         }
aucs = list()

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    start_time = time()

    print('Training on fold {}'.format(fold + 1))

    

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

    

    aucs.append(clf.best_score['valid_1']['auc'])

    

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

print('-' * 30)

print('Training has finished.')

print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))

print('Mean AUC:', np.mean(aucs))

print('-' * 30)
best_iter = clf.best_iteration

clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)

clf.fit(X, y)
sub = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

sub['isFraud'] = clf.predict_proba(X_test)[:, 1]

sub.to_csv('lgbm_v4_submission.csv', index=False)