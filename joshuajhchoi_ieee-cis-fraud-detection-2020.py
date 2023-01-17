# importing libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from IPython.display import display
pd.options.display.precision = 15
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.max_colwidth = 1000

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
from numba import jit
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb

%load_ext autoreload
%autoreload 2

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

seed = 42
seed_everything(seed)
folder_path = '../input/'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')
sub = pd.read_csv(f'{folder_path}sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
train_identity.head()
train_identity.shape
train_transaction.head()
train_transaction.shape
test_identity.head()
test_transaction.head()
# 어떤 이유에선지 테스트 컬럼에서 언더바 대신 하이픈으로 쓴 것이 있어서 좀 바꿉니다.
test_identity.columns = [col.replace('-', '_') for col in test_identity.columns]

# merge method를 사용하여 트레인과 테스트의 트랜잭션(거래)와 아이덴티티(신분)을 파일을 합합니다. 
# 여기서 on은 그 것을 인덱스로 합치라는 것이고 left나 right는 왼쪽 것 기준으로 또는 오른쪽 것 기준으로 파일을 정열하는데...여기서 말하는 기준은 데이터 값이 있는 것입니다.
# 다시 말해 left로 되어 있는데 왼쪽의 transaction의 해당 행이 빈 값이면 identity에 값이 있어도 그 행은 보이지 않게되고, 반대로 right라면 그 행은 보이게 됩니다.

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_transaction, train_identity, test_transaction, test_identity
test[[f'C{i}' for i in range(1,15)]] = test[[f'C{i}' for i in range(1,15)]].fillna(0)
end_mem = train.memory_usage().sum() / 1024 ** 2 + test.memory_usage().sum() / 1024 ** 2
print(f'Mem. usage {end_mem:5.2f} Mb')
def reduce_mem_usage(memory_df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = memory_df.memory_usage().sum() / 1024 ** 2
    for col in memory_df.columns:
        col_type = memory_df[col].dtypes
        if col_type in numerics:
            c_min = memory_df[col].min()
            c_max = memory_df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    memory_df[col] = memory_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    memory_df[col] = memory_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    memory_df[col] = memory_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    memory_df[col] = memory_df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    memory_df[col] = memory_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    memory_df[col] = memory_df[col].astype(np.float32)
                else:
                    memory_df[col] = memory_df[col].astype(np.float64)
    end_mem = memory_df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return memory_df
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train.head()
plt.hist(train['id_01'], bins=77);
plt.title('Distribution of id_01 variable');
train['id_03'].value_counts(dropna=False, normalize=True).head()
train['id_11'].value_counts(dropna=False, normalize=True).head()
plt.hist(train['id_07']);
plt.title('Distribution of id_07 variable');
plt.hist(train['TransactionDT'], label='train');
plt.hist(test['TransactionDT'], label='test');
plt.legend();
plt.title('Distribution of transactiond dates');
start_date = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
print (start_date)
for col in train.columns:
    if 'card' in col:
        print(f"{col} has {train[col].isnull().sum()} missing values.")
group_with_mode = train.groupby(['card1']).agg({'card2': ['nunique', pd.Series.mode]})
group_with_mode.head(10)
non_one = group_with_mode[group_with_mode['card2']['nunique'] > 1].shape[0]
card1_nunique = train['card1'].nunique()
print(f'Number of unique values of card1: {card1_nunique}')
print(f'Number of unique values of card1 which have more than one unique value: {non_one}')
for card in ['card2','card3','card4','card5','card6']:
    group_with_mode = train.groupby(['card1']).agg({card: ['nunique', pd.Series.mode]})
    to_merge = group_with_mode[group_with_mode[card]['nunique'] == 1][card]['mode']
    merged = pd.merge(train['card1'], to_merge, on='card1', how='left')
    merged['mode'] = merged['mode'].fillna(train[card])
    train[card] = merged['mode']
    
    group_with_mode = test.groupby(['card1']).agg({card: ['nunique', pd.Series.mode]})
    to_merge = group_with_mode[group_with_mode[card]['nunique'] == 1][card]['mode']
    merged = pd.merge(test['card1'], to_merge, on='card1', how='left')
    merged['mode'] = merged['mode'].fillna(test[card])
    test[card] = merged['mode']
for df in [train,test]:
    df['DaysFromStart'] = np.floor(df['TransactionDT']/(60*60*24)) - 1
    df['D1-DaysFromStart'] = df['D1'] - df['DaysFromStart']
for df in [train,test]:
    df['uid'] = df['ProductCD'].astype(str) + '_' + df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid'] = df['uid'] + '_' + df['card3'].astype(str) + '_' + df['card4'].astype(str)
    df['uid'] = df['uid'] + '_' + df['card5'].astype(str) + '_' + df['card6'].astype(str)
    df['uid'] = df['uid'] + '_' + df['addr1'].astype(str) + '_' + df['D1-DaysFromStart'].astype(str)
len(set(train['uid']).intersection(set(test['uid']))), train.uid.nunique(), test.uid.nunique()
train['uid'].value_counts(dropna=False, normalize=True).head(20)
train['DeviceInfo'].value_counts(dropna=False, normalize=True).head(20)
train['DeviceInfo'].nunique()
for df in [train, test]:
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
df['DeviceInfo_device'].value_counts(dropna=False, normalize=True).head(20)
df['DeviceInfo_version'].value_counts(dropna=False, normalize=True).head(20)
df['id_30'].value_counts(dropna=False, normalize=True).head(20)
train['id_30'].nunique()
df['id_31'].value_counts(dropna=False, normalize=True).head(20)
for df in [train, test]:
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
df['id_30_device'].value_counts(dropna=False, normalize=True).head(20)
train.D1.value_counts()
train.D2.value_counts()
train.D3.value_counts()
train.D4.value_counts()
train.D5.value_counts()
train.D6.value_counts()
train.D7.value_counts()
train.D8.value_counts()
train.D9.value_counts()
train.D10.value_counts()
train.D11.value_counts()
train.D12.value_counts()
train.D13.value_counts()
train.D14.value_counts()
train.D15.value_counts()
plt.hist(train['D8']);
plt.title('Distribution of D8 variable');
plt.hist(train['D9']);
plt.title('Distribution of D9 variable');
for df in [train, test]:
    # 클립 메소드를 사용하면 음수를 0으로 만들 수 있습니다
    for col in ['D'+str(i) for i in range(1,16)]:
        df[col] = df[col].clip(0)
    
    # D9을 빈 값이 아닌 것을 따로 떼어내어 D9_not_na 항목을 새로 만듭니다.
    df['D9_not_na'] = np.where(df['D9'].isna(),0,1)
    # D8을 1보다 크거나 같은 것만 떼어내어 D8_not_same_day 항목을 만듭니다
    df['D8_not_same_day'] = np.where(df['D8']>=1,1,0)
    # D8에서 소수점 이하를 뽑아서 D8_D9_decimal_dist로 정의한 후 여기서 D9을 뺍니다
    df['D8_D9_decimal_dist'] = df['D8'].fillna(0)-df['D8'].fillna(0).astype(int)
    df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist']-df['D9'])**2)**0.5
    df['D8'] = df['D8'].fillna(-1).astype(int)
df.D9_not_na.value_counts()
df.D8_not_same_day.value_counts()
df.D8_D9_decimal_dist.value_counts()
df.D8.value_counts()
plt.hist(train['TransactionAmt']);
plt.title('Distribution of TransactionAmt variable');
np.percentile(train['TransactionAmt'], 99)
# 클립하여 0부터 5000사이의 값으로 봅니다
train['TransactionAmt'] = train['TransactionAmt'].clip(0,5000)
test['TransactionAmt']  = test['TransactionAmt'].clip(0,5000)

# 거래 액수가 일반적인지 아닌지를 봅니다
train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)
train.TransactionAmt_check.value_counts()
test.TransactionAmt_check.value_counts()
for df in [train, test]:
    df['ProductCD_card1'] = df['ProductCD'].astype(str) + '_' + df['card1'].astype(str)
    df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    df['TransactionAmt_dist2'] = df['TransactionAmt'].astype(str) + '_' + df['dist2'].astype(str)
    df['card3_card5'] = df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['ProductCD_TransactionAmt'] = df['ProductCD'].astype(str) + '_' + df['TransactionAmt'].astype(str)
    df['cents'] = np.round(df['TransactionAmt'] - np.floor(df['TransactionAmt']), 3)
    df['ProductCD_cents'] = df['ProductCD'].astype(str) + '_' + df['cents'].astype(str)
    df['TransactionAmt'] = np.log1p(df['TransactionAmt'])
df.ProductCD_card1.value_counts()
df.card1_addr1.value_counts()
df.TransactionAmt_dist2.value_counts()
df.ProductCD_TransactionAmt.value_counts()
df.TransactionAmt.value_counts()
test['ProductCD_card1']
# 아래 항목들은 다른 것을 만드는데만 유용하니 이제 없애도록 하겠습니다
train = train.drop(['DaysFromStart','D1-DaysFromStart'], axis=1)
test = test.drop(['DaysFromStart','D1-DaysFromStart'], axis=1)
for df in [train, test]:
    # 집계를위한 임시 항목들
    df['DT'] = df['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))
    df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
    df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear).astype(np.int8)
    df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)

    df['DT_hour'] = df['DT'].dt.hour.astype(np.int8)
    df['DT_day_week'] = df['DT'].dt.dayofweek.astype(np.int8)
    df['DT_day_month'] = df['DT'].dt.day.astype(np.int8)

    # 잠재적 솔로 항목
    df['is_december'] = df['DT'].dt.month
    df['is_december'] = (df['is_december'] == 12).astype(np.int8)
y = train['isFraud']
X = train.drop(['isFraud'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
def freq_encode_full(df1, df2, col, normalize=True):
    """
    Encode

    https://www.kaggle.com/cdeotte/high-scoring-lgbm-malware-0-702-0-775
    """
    df = pd.concat([df1[col], df2[col]])
    freq_dict = df.value_counts(dropna=False, normalize=normalize).to_dict()
    col_name = col + '_freq_enc_full'
    return col_name, freq_dict
def process_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # 제자리에서 변경되지 않도록 복사하십시오.
    train_df = df_train.copy()
    test_df = df_test.copy()
    remove_features = ['TransactionID', 'TransactionDT', 'DT', 'DT_M', 'DT_W', 'DT_D', 'DT_hour', 'DT_day_week',
                       'DT_day_month',
                       'ProductCD_card1', 'card1_addr1', 'TransactionAmt_dist2', 'card3_card5', 'uid',
                       'D5_DT_W_std_score',
                       'ProductCD_TransactionAmt_DT_W', 'D4_DT_D_std_score', 'D15_DT_D_std_score', 'D3_DT_W_std_score',
                       'D11_DT_W_std_score',
                       'card3_card5_DT_W_week_day_dist', 'card5_DT_W_week_day_dist', 'D10_DT_D_std_score',
                       'card3_card5_DT_D', 'ProductCD_cents_DT_D',
                       'D4_DT_W_std_score', 'D15_DT_W_std_score', 'uid_DT_D', 'card3_DT_W_week_day_dist',
                       'D10_DT_W_std_score', 'D8_DT_D_std_score',
                       'card3_card5_DT_W', 'ProductCD_cents_DT_W', 'uid_DT_W', 'D8_DT_W_std_score',
                       'ProductCD_TransactionAmt']

    """
     다음 코드는 다음을 수행합니다.
     * 학습 및 테스트 데이터에서 특정 열의 값을 결합하고 'value_counts'계산
     *`valid_card`를 하나 이상의 값이 있는 카테고리로 정의
     * 트레인 데이터의 열 범주도 테스트 데이터에 있으면 유지하고, 그렇지 않으면 None으로 바꿉니다.
     * 트레인 데이터의 열 범주가`valid_card`에 있으면 유지하고, 그렇지 않으면 None으로 바꿉니다.
     * 테스트 데이터에 대해 동일하게 수행
    """
    for col in ['card1', 'ProductCD_card1', 'card1_addr1', 'TransactionAmt_dist2']:
        valid_card = pd.concat([train_df[[col]], test_df[[col]]])
        valid_card = valid_card[col].value_counts()

        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
        train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)

        test_df[col] = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

    # 이전과 같이 값이 하나의 데이터 세트에만있는 경우 None으로 바꿉니다
    for col in ['card2', 'card3', 'card4', 'card5', 'card6']:
        train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

    ####### 투레인 데이터에서 지난달의 최대 값으로 C 열을 자릅니다.
    i_cols = ['C' + str(i) for i in range(1, 15)]
    for df in [train_df, test_df]:
        for col in i_cols:
            max_value = train_df[train_df['DT_M'] == train_df['DT_M'].max()][col].max()
            df[col] = df[col].clip(None, max_value)

    ####### V feature - NaN group agg
    # null 값과 열 목록을 포함하는 dictionary가 됩니다.
    nans_groups = {}
    nans_df = pd.concat([train_df, test_df]).isna()

    i_cols = ['V' + str(i) for i in range(1, 340)]
    for col in i_cols:
        # NaN 값을 세워 봅니다
        cur_group = nans_df[col].sum()
        if cur_group > 0:
            try:
                nans_groups[cur_group].append(col)
            except:
                nans_groups[cur_group] = [col]

    for i, (n_group, n_cols) in enumerate(nans_groups.items()):
        for df in [train_df, test_df]:
            df[f'nan_group_{i}_sum'] = df[n_cols].sum(axis=1)
            df[f'nan_group_{i}_mean'] = df[n_cols].mean(axis=1)
            df[f'nan_group_{i}_std'] = df[n_cols].std(axis=1)

    del nans_groups, nans_df
    remove_features += i_cols
    # 너무나 많은 공간을 차지합니다. 필요한 것만 남기고 drop 시킵니다.
    i_cols = [i for i in i_cols if i not in ['V258', 'V306', 'V307', 'V308', 'V294']]
    train_df = train_df.drop(i_cols, axis=1)
    test_df = test_df.drop(i_cols, axis=1)

    # frequency encoding. 인코딩한 항목과 오리지날 항목을 `remove_features`에 더합니다
    i_cols = [
        'ProductCD_TransactionAmt', 'ProductCD_cents', 'cents',
        'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
        'id_30', 'id_30_device', 'id_30_version',
        'id_31', 'id_31_device',
        'id_33',
    ]
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

        remove_features.append(col)

    # 원래 항목을 유지하면서 frequency encoding을 합니다. 
    i_cols = ['id_01', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_13',
              'id_14', 'id_17', 'id_18', 'id_19', 'id_20',
              'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'card1', 'card2', 'card3', 'card5', 'ProductCD_card1',
              'card1_addr1', 'TransactionAmt_dist2']
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

    return train_df, test_df, remove_features
%%time
X_train, X_val, remove_features = process_data(X_train, X_val)
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, remove_features: list):
    train_df = df_train.copy()
    test_df = df_test.copy()

    # Label Encoding    
    for f in train_df.columns:
        if train_df[f].dtype == 'object' or test_df[f].dtype == 'object':
            train_df[f] = train_df[f].fillna('unseen_before_label')
            test_df[f] = test_df[f].fillna('unseen_before_label')
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            train_df[f] = train_df[f].astype('category')
            test_df[f] = test_df[f].astype('category')

    print('remove_features:', remove_features)

    feature_columns = [col for col in list(train_df) if col not in remove_features]
    categorical_features = [col for col in feature_columns if train_df[col].dtype.name == 'category']
    categorical_features = [col for col in categorical_features if col not in remove_features]

    print(f'train.shape : {train_df[feature_columns].shape}, test.shape : {test_df[feature_columns].shape}')

    return train_df[feature_columns], test_df[feature_columns], categorical_features

X_train, X_val, categorical_features = feature_engineering(X_train, X_val, remove_features)
lgb_params = {
    'objective': 'binary',
    'metric': 'None',
    'learning_rate': 0.01,
    'num_leaves': 2**8,
    'max_bin': 255,
    'max_depth': -1,
    'bagging_freq': 5,
    'bagging_fraction': 0.7,
    'bagging_seed': seed,
    'feature_fraction': 0.7,
    'feature_fraction_seed': seed,
    'first_metric_only': True,
    'verbose': 100,
    'n_jobs': -1,
    'seed': seed,
}
@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_pred, y_true):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true.get_label(), y_pred), True

def make_val_prediction(X_train, y_train, X_val, y_val, seed=0, seed_range=3, lgb_params=None,
                        category_cols=None):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    auc_scores = []
    best_iterations = []
    val_preds = np.zeros((X_val.shape[0], 3))

    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = X_train.columns.tolist()
    feature_importance_df['gain_importance'] = 0

    for i, s in enumerate(range(seed, seed + seed_range)):
        seed_everything(s)
        params = lgb_params.copy()
        params['seed'] = s
        params['bagging_seed'] = s
        params['feature_fraction_seed'] = s

        clf = lgb.train(params, train_data, 10000, valid_sets=[train_data, val_data],
                        categorical_feature=categorical_features,
                        early_stopping_rounds=500, feval=eval_auc, verbose_eval=200)

        best_iteration = clf.best_iteration
        best_iterations.append(best_iteration)
        val_pred = clf.predict(X_val, best_iteration)
        val_preds[:, i] = val_pred

        auc = fast_auc(y_val, val_pred)
        auc_scores.append(auc)
        print('seed:', s, ', auc:', auc, ', best_iteration:', best_iteration)

        feature_importance_df['gain_importance'] += clf.feature_importance('gain') / seed_range

    auc_scores = np.array(auc_scores)
    best_iterations = np.array(best_iterations)
    best_iteration = int(np.mean(best_iterations))

    avg_pred_auc = fast_auc(y_val, np.mean(val_preds, axis=1))
    print(
        f'avg pred auc: {avg_pred_auc:.5f}, avg auc: {np.mean(auc_scores):.5f}+/-{np.std(auc_scores):.5f}, avg best iteration: {best_iteration}')

    feature_importance_df = feature_importance_df.sort_values(by='gain_importance', ascending=False).reset_index(
        drop=True)
    plt.figure(figsize=(16, 12));
    sns.barplot(x="gain_importance", y="feature", data=feature_importance_df[:50])
    plt.title('LGB Features (avg over folds)');

    return feature_importance_df, best_iteration, val_preds
feature_importance_df, best_iteration, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=categorical_features,
                                                       lgb_params=lgb_params)
def prediction(X, y, X_test, best_iteration, seed=seed, category_cols=None, n_folds=5):
    print('best iteration:', best_iteration)
    preds = np.zeros((X_test.shape[0], n_folds))

    print(X.shape, X_test.shape)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for i, (trn_idx, _) in enumerate(skf.split(X, y)):
        fold = i + 1

        tr_x, tr_y = X.iloc[trn_idx, :], y.iloc[trn_idx]

        tr_data = lgb.Dataset(tr_x, label=tr_y)

        clf = lgb.train(lgb_params, tr_data, best_iteration, categorical_feature=category_cols)
        preds[:, i] = clf.predict(X_test)

    return preds

%%time
X_full, X_test, remove_features = process_data(X, test.copy())
X_full, X_test, categorical_features = feature_engineering(X_full, X_test, remove_features)
preds = prediction(X_full, y, X_test, best_iteration, category_cols=categorical_features)
sub['isFraud'] = np.mean(preds, axis=1)
sub.to_csv('sub_1.csv', index=False)
def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col + '_' + period

            dt_df[col] = dt_df[col].astype(float)

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
            temp_mean.index = temp_mean[period].values
            temp_mean = temp_mean['mean'].to_dict()

            temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
            temp_std.index = temp_std[period].values
            temp_std = temp_std['std'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)
            dt_df['temp_mean'] = dt_df[period].map(temp_mean)
            dt_df['temp_std'] = dt_df[period].map(temp_std)

            dt_df[new_col + '_min_max'] = (dt_df[col] - dt_df['temp_min']) / (dt_df['temp_max'] - dt_df['temp_min'])
            dt_df[new_col + '_std_score'] = (dt_df[col] - dt_df['temp_mean']) / (dt_df['temp_std'])
            del dt_df['temp_min'], dt_df['temp_max'], dt_df['temp_mean'], dt_df['temp_std']
    return dt_df
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, remove_features: list):
    # 변경되지 않도록 복사합니다
    train_df = df_train.copy()
    test_df = df_test.copy()

    i_cols = ['D' + str(i) for i in range(1, 16)]

    ####### Values Normalization
    i_cols.remove('D1')
    i_cols.remove('D2')
    i_cols.remove('D9')
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, i_cols)

    # TransactionAmt Normalization
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, ['TransactionAmt'])

    # normalize by max train value
    for col in ['D1', 'D2']:
        for df in [train_df, test_df]:
            df[col + '_scaled'] = df[col] / train_df[col].max()

    # frequency encoding
    i_cols = ['D' + str(i) for i in range(1, 16)] + ['DT_D', 'DT_W', 'ProductCD_TransactionAmt', 'ProductCD_cents']
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

        remove_features.append(col)

    # Label Encoding    
    for f in train_df.columns:
        if train_df[f].dtype == 'object' or test_df[f].dtype == 'object':
            train_df[f] = train_df[f].fillna('unseen_before_label')
            test_df[f] = test_df[f].fillna('unseen_before_label')
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            train_df[f] = train_df[f].astype('category')
            test_df[f] = test_df[f].astype('category')

    print('remove_features:', remove_features)
    print(f'train.shape : {train_df.shape}, test.shape : {test_df.shape}')

    ########################### Final features list
    feature_columns = [col for col in list(train_df) if col not in remove_features]
    print('feature_columns:', len(feature_columns))
    categorical_features = [col for col in feature_columns if train_df[col].dtype.name == 'category']
    categorical_features = [col for col in categorical_features if col not in remove_features]

    return train_df[feature_columns], test_df[feature_columns], categorical_features
%%time
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, remove_features = process_data(X_train, X_val)
X_train, X_val, categorical_features = feature_engineering(X_train, X_val, remove_features)
feature_importance_df, best_iteration, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=categorical_features,
                                                       lgb_params=lgb_params)
%%time
X_full, X_test, remove_features = process_data(X, test.copy())
X_full, X_test, categorical_features = feature_engineering(X_full, X_test, remove_features)
preds = prediction(X_full, y, X_test, best_iteration, category_cols=categorical_features)
sub['isFraud'] = np.mean(preds, axis=1)
sub.to_csv('sub_2.csv', index=False)
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, remove_features: list):
    # 원본이 변경되지 않도록 복사해 사용합니다
    train_df = df_train.copy()
    test_df = df_test.copy()

    i_cols = ['D' + str(i) for i in range(1, 16)]

    ####### Values Normalization
    i_cols.remove('D1')
    i_cols.remove('D2')
    i_cols.remove('D9')
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, i_cols)

    # TransactionAmt Normalization
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, ['TransactionAmt'])

    # normalize by max train value
    for col in ['D1', 'D2']:
        for df in [train_df, test_df]:
            df[col + '_scaled'] = df[col] / train_df[col].max()

    # frequency encoding
    i_cols = ['D' + str(i) for i in range(1, 16)] + ['DT_D', 'DT_W', 'ProductCD_TransactionAmt', 'ProductCD_cents']
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

        remove_features.append(col)

    i_cols = ['C' + str(i) for i in range(1, 15)]
    # 여기서는 C column의 0들을 카운트합니다
    for df in [train_df, test_df]:
        df['c_cols_0_bin'] = ''
        for c in i_cols:
            df['c_cols_0_bin'] += (df[c] == 0).astype(int).astype(str)
    col_name, freq_dict = freq_encode_full(train_df, test_df, 'c_cols_0_bin')

    train_df[col_name] = train_df['c_cols_0_bin'].map(freq_dict).astype('float32')
    test_df[col_name] = test_df['c_cols_0_bin'].map(freq_dict).astype('float32')

    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

    # Label Encoding    
    for f in train_df.columns:
        if train_df[f].dtype == 'object' or test_df[f].dtype == 'object':
            train_df[f] = train_df[f].fillna('unseen_before_label')
            test_df[f] = test_df[f].fillna('unseen_before_label')
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            train_df[f] = train_df[f].astype('category')
            test_df[f] = test_df[f].astype('category')

    print('remove_features:', remove_features)
    print(f'train.shape : {train_df.shape}, test.shape : {test_df.shape}')

    ########################### Final features list
    feature_columns = [col for col in list(train_df) if col not in remove_features]
    print('feature_columns:', len(feature_columns))
    categorical_features = [col for col in feature_columns if train_df[col].dtype.name == 'category']
    categorical_features = [col for col in categorical_features if col not in remove_features]

    return train_df[feature_columns], test_df[feature_columns], categorical_features
%%time
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, remove_features = process_data(X_train, X_val)
X_train, X_val, categorical_features = feature_engineering(X_train, X_val, remove_features)
feature_importance_df, best_iteration, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=categorical_features,
                                                       lgb_params=lgb_params)
%%time
X_full, X_test, remove_features = process_data(X, test.copy())
X_full, X_test, categorical_features = feature_engineering(X_full, X_test, remove_features)
preds = prediction(X_full, y, X_test, best_iteration, category_cols=categorical_features)
sub['isFraud'] = np.mean(preds, axis=1)
sub.to_csv('sub_3.csv', index=False)
def uid_aggregation(train_df, test_df, main_columns, uids, aggregations):
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col + '_' + main_column + '_' + agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name] = test_df[col].map(temp_df)
    return train_df, test_df
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, remove_features: list):
    # 복사해 둡니다
    train_df = df_train.copy()
    test_df = df_test.copy()

    i_cols = ['D' + str(i) for i in range(1, 16)]

    ####### Values Normalization
    i_cols.remove('D1')
    i_cols.remove('D2')
    i_cols.remove('D9')
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, i_cols)

    # TransactionAmt Normalization
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, ['TransactionAmt'])

    for col in ['D1', 'D2']:
        for df in [train_df, test_df]:
            df[col + '_scaled'] = df[col] / train_df[col].max()

    i_cols = ['D' + str(i) for i in range(1, 16)] + ['DT_D', 'DT_W', 'ProductCD_TransactionAmt', 'ProductCD_cents']
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

        remove_features.append(col)

    i_cols = ['C' + str(i) for i in range(1, 15)]

    for df in [train_df, test_df]:
        df['c_cols_0_bin'] = ''
        for c in i_cols:
            df['c_cols_0_bin'] += (df[c] == 0).astype(int).astype(str)
    col_name, freq_dict = freq_encode_full(train_df, test_df, 'c_cols_0_bin')

    train_df[col_name] = train_df['c_cols_0_bin'].map(freq_dict).astype('float32')
    test_df[col_name] = test_df['c_cols_0_bin'].map(freq_dict).astype('float32')

    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

    i_cols = ['TransactionAmt', 'id_01', 'id_02', 'id_05', 'id_06', 'id_09', 'id_14', 'dist1'] + ['C' + str(i) for i in
                                                                                                  range(1, 15)]
    uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'card3_card5']
    aggregations = ['mean', 'std']

    # uIDs aggregations
    train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

    i_cols = [
                 'V258',
                 'V306', 'V307', 'V308', 'V294'
             ] + ['D' + str(i) for i in range(1, 16)]
    uids = ['uid', 'card3_card5']
    aggregations = ['mean', 'std']
    train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

    # Label Encoding    
    for f in train_df.columns:
        if train_df[f].dtype == 'object' or test_df[f].dtype == 'object':
            train_df[f] = train_df[f].fillna('unseen_before_label')
            test_df[f] = test_df[f].fillna('unseen_before_label')
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            train_df[f] = train_df[f].astype('category')
            test_df[f] = test_df[f].astype('category')

    print('remove_features:', remove_features)
    print(f'train.shape : {train_df.shape}, test.shape : {test_df.shape}')

    ########################### Final features list
    feature_columns = [col for col in list(train_df) if col not in remove_features]
    print('feature_columns:', len(feature_columns))
    categorical_features = [col for col in feature_columns if train_df[col].dtype.name == 'category']
    categorical_features = [col for col in categorical_features if col not in remove_features]

    return train_df[feature_columns], test_df[feature_columns], categorical_features
%%time
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, remove_features = process_data(X_train, X_val)
X_train, X_val, categorical_features = feature_engineering(X_train, X_val, remove_features)
feature_importance_df, best_iteration, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=categorical_features,
                                                       lgb_params=lgb_params)
%%time
X_full, X_test, remove_features = process_data(X, test.copy())
X_full, X_test, categorical_features = feature_engineering(X_full, X_test, remove_features)
preds = prediction(X_full, y, X_test, best_iteration, category_cols=categorical_features)
sub['isFraud'] = np.mean(preds, axis=1)
sub.to_csv('sub_4.csv', index=False)
def timeblock_frequency_encoding(train_df, test_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = col + '_' + period
            print('timeblock frequency encoding:', new_col)
            train_df[new_col] = train_df[col].astype(str) + '_' + train_df[period].astype(str)
            test_df[new_col] = test_df[col].astype(str) + '_' + test_df[period].astype(str)

            temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
            fq_encode = temp_df[new_col].value_counts(normalize=True).to_dict()

            train_df[new_col] = train_df[new_col].map(fq_encode)
            test_df[new_col] = test_df[new_col].map(fq_encode)

            train_df[new_col] = train_df[new_col] / train_df[period + '_freq_enc_full']
            test_df[new_col] = test_df[new_col] / test_df[period + '_freq_enc_full']

    return train_df, test_df
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame, remove_features: list):
    # 복사해 둡니다
    train_df = df_train.copy()
    test_df = df_test.copy()

    i_cols = ['D' + str(i) for i in range(1, 16)]

    ####### Values Normalization
    i_cols.remove('D1')
    i_cols.remove('D2')
    i_cols.remove('D9')
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, i_cols)

    # TransactionAmt Normalization
    periods = ['DT_D', 'DT_W']
    for df in [train_df, test_df]:
        df = values_normalization(df, periods, ['TransactionAmt'])

    for col in ['D1', 'D2']:
        for df in [train_df, test_df]:
            df[col + '_scaled'] = df[col] / train_df[col].max()

    i_cols = ['D' + str(i) for i in range(1, 16)] + ['DT_D', 'DT_W', 'ProductCD_TransactionAmt', 'ProductCD_cents']
    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

        remove_features.append(col)

    i_cols = ['C' + str(i) for i in range(1, 15)]

    for df in [train_df, test_df]:
        df['c_cols_0_bin'] = ''
        for c in i_cols:
            df['c_cols_0_bin'] += (df[c] == 0).astype(int).astype(str)
    col_name, freq_dict = freq_encode_full(train_df, test_df, 'c_cols_0_bin')

    train_df[col_name] = train_df['c_cols_0_bin'].map(freq_dict).astype('float32')
    test_df[col_name] = test_df['c_cols_0_bin'].map(freq_dict).astype('float32')

    for col in i_cols:
        col_name, freq_dict = freq_encode_full(train_df, test_df, col)

        train_df[col_name] = train_df[col].map(freq_dict).astype('float32')
        test_df[col_name] = test_df[col].map(freq_dict).astype('float32')

    i_cols = ['TransactionAmt', 'id_01', 'id_02', 'id_05', 'id_06', 'id_09', 'id_14', 'dist1'] + ['C' + str(i) for i in
                                                                                                  range(1, 15)]
    uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'card3_card5']
    aggregations = ['mean', 'std']

    # uIDs aggregations
    train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

    i_cols = [
                 'V258',
                 'V306', 'V307', 'V308', 'V294'
             ] + ['D' + str(i) for i in range(1, 16)]
    uids = ['uid', 'card3_card5']
    aggregations = ['mean', 'std']
    train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

    i_cols = ['ProductCD_TransactionAmt', 'ProductCD_cents', 'ProductCD_cents', 'uid', 'card3_card5']
    periods = ['DT_D', 'DT_W']
    train_df, test_df = timeblock_frequency_encoding(train_df, test_df, periods, i_cols)

    # Label Encoding    
    for f in train_df.columns:
        if train_df[f].dtype == 'object' or test_df[f].dtype == 'object':
            train_df[f] = train_df[f].fillna('unseen_before_label')
            test_df[f] = test_df[f].fillna('unseen_before_label')
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            train_df[f] = train_df[f].astype('category')
            test_df[f] = test_df[f].astype('category')

    print('remove_features:', remove_features)
    print(f'train.shape : {train_df.shape}, test.shape : {test_df.shape}')

    ########################### Final features list
    feature_columns = [col for col in list(train_df) if col not in remove_features]
    print('feature_columns:', len(feature_columns))
    categorical_features = [col for col in feature_columns if train_df[col].dtype.name == 'category']
    categorical_features = [col for col in categorical_features if col not in remove_features]

    return train_df[feature_columns], test_df[feature_columns], categorical_features

%%time
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, remove_features = process_data(X_train, X_val)
X_train, X_val, categorical_features = feature_engineering(X_train, X_val, remove_features)
feature_importance_df, best_iteration, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=categorical_features,
                                                       lgb_params=lgb_params)
%%time
X_full, X_test, remove_features = process_data(X, test.copy())
X_full, X_test, categorical_features = feature_engineering(X_full, X_test, remove_features)
preds = prediction(X_full, y, X_test, best_iteration, category_cols=categorical_features)
sub['isFraud'] = np.mean(preds, axis=1)
sub.to_csv('sub_5.csv', index=False)