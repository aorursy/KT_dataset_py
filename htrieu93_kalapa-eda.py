# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
file_path = '/kaggle/input/kalapa-credit/Kalapa_s Credit Scoring challenge/'
train = pd.read_csv(file_path + 'train.csv')
test = pd.read_csv(file_path + 'test.csv')
df = pd.concat([train, test], sort=False)
df.reset_index(drop=True, inplace=True)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

rep_dict = {'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ặ': 'a', 'ẳ': 'a', 'ẵ': 'a',
            'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ậ': 'a', 'ẩ': 'a', 'ẫ': 'a',
            'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ợ': 'o', 'ở': 'o', 'ỡ': 'o',
            'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ộ': 'o', 'ổ': 'o', 'ỗ': 'o',
            'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ự': 'u', 'ử': 'u', 'ữ': 'u',
            'ê': 'e', 'ế': 'e', 'ề': 'e', 'ệ': 'e', 'ể': 'e', 'ễ': 'e',
            'ì': 'i', 'í': 'i', 'ị': 'i', 'ỉ': 'i', 'ĩ': 'i',
            'à': 'a', 'á': 'a', 'ạ': 'a', 'ả': 'a', 'ã': 'a',
            'ò': 'o', 'ó': 'o', 'ọ': 'o', 'ỏ': 'o', 'õ': 'o',
            'ù': 'u', 'ú': 'u', 'ụ': 'u', 'ủ': 'u', 'ũ': 'u', 
            'è': 'e', 'é': 'e', 'ẹ': 'e', 'ẻ': 'e', 'ẽ': 'e', 
            'ỳ': 'y', 'ý': 'y', 'ỵ': 'y', 'ỷ': 'y', 'ỹ': 'y', 
            'đ': 'd'}

def quick_analysis(col):
    train_gb = train.loc[~train[col].isnull(), [col, 'label']].groupby(col).agg(['mean', 'sum', 'count'])
    train_gb.columns = train_gb.columns.droplevel(0)
    train_gb.rename(columns={'mean':'Mean target', 'sum':'Sum target', 'count':'Count in Train'}, inplace=True)
    
    test_count = test.loc[~test[col].isnull(), col].value_counts().to_frame()
    test_count.rename(columns={col:'Count in Test'}, inplace=True)
    
    train_gb = train_gb.merge(test_count, left_index=True, right_index=True, how='outer')
    
    train_null = train.loc[train[col].isnull(), 'label'].agg(['mean', 'sum', 'count']).to_frame().transpose()
    train_null.rename(columns={'mean':'Mean target', 'sum':'Sum target', 'count':'Count in Train'}, inplace=True)
    train_null['Count in Test'] = test[col].isnull().sum()
    train_null.rename(index={'label':'NaN'}, inplace=True)
    
    train_gb = pd.concat([train_gb, train_null])
    return(train_gb)

def clean_field_7():
    df['FIELD_7'] = df.loc[~df['FIELD_7'].isnull(), 'FIELD_7'].apply(lambda x: re.findall(r"'([^']*)'", x))
    
    max_col = df.loc[~df['FIELD_7'].isnull(), 'FIELD_7'].apply(lambda x: len(x)).max()
    df['FIELD_7_count'] = df.loc[~df['FIELD_7'].isnull(), 'FIELD_7'].apply(lambda x: len(x))
    for i in range(max_col):
        df['FIELD_7_' + str(i)] = df.loc[~df['FIELD_7'].isnull(), 'FIELD_7'].apply(lambda x: x[i] if i < len(x) else np.nan)
    df['FIELD_7_nunique'] = df[['FIELD_7_0', 'FIELD_7_1', 'FIELD_7_2', 'FIELD_7_3', 'FIELD_7_4', 'FIELD_7_5', 'FIELD_7_6',
                                'FIELD_7_7', 'FIELD_7_8', 'FIELD_7_9', 'FIELD_7_10', 'FIELD_7_11', 'FIELD_7_12', 'FIELD_7_13']].nunique()

def clean_field_9():
    df['FIELD_9'].replace('na|74|79|80|86|CK|MS|NO|TL', np.nan, inplace=True)  
        
def clean_field_11():
    df['FIELD_11'].replace('None', -1, regex=True, inplace=True)
#     df['FIELD_11'].fillna(-999, inplace=True)
    df['FIELD_11'] = df['FIELD_11'].astype(np.float32)

def clean_field_12():
    df['FIELD_12'].replace('DK|DN|DT|GD|HT|TN|XK', 0, regex=True, inplace=True)
    df['FIELD_12'].replace('None', -1, regex=True, inplace=True)
#     df['FIELD_12'].fillna(-999, inplace=True)
    df['FIELD_12'] = df['FIELD_12'].astype(np.float32)
    
def vn_to_eng(x):
    x = x.lower()
    return(''.join([rep_dict.get(c, c) for c in x]))

# Map Vnese to Eng
def vn_to_eng_col(col):
#     df[col].fillna('missing', inplace=True)
    df[col] = df.loc[~df[col].isnull(), col].apply(lambda s: vn_to_eng(s))      
        
def sync_values(col, fix_na=True):
    if fixed_na:
        df.loc[~df[col].isnull() & df[col].str.contains('None'), col] = np.nan
    df.loc[~df[col].isnull(), col] = df.loc[~df[col].isnull(), col].astype(np.int32)

def fixed_na(cols):
    for col in cols:
        df.loc[~df[col].isnull() & df[col].str.contains('None|na|missing|none'), col] = 'None'
        df.loc[~df[col].isnull() & (df[col] == -1), col] = np.nan

def encode_NAN(cols):
    for col in cols:
        df[col].fillna(-999, inplace=True)
        
def encode_NONE(cols):
    for col in cols:
        df[col].replace('None', -1, regex=True, inplace=True)
    
def count_none_values(cols, grp):
    for col in cols:
        df.loc[df[col] == 'None', 'count_none_' + str(col)] = 1
    df['count_none_' + str(grp)] = df.loc[:, df.columns.str.contains('count_none')].sum(axis=1, skipna=True)
        
# Boolean encoding
def encode_BL(cols, fix_na=False):
    for col in cols:
        df[col].replace('True|TRUE', 1, regex=True, inplace=True)
        df[col].replace('False|FALSE', 0, regex=True, inplace=True)
        df[col].replace('None', -1, regex=True, inplace=True)
        if fix_na:
            df.loc[df[col].isnull(), col] = -999
        df[col] = df[col].astype(np.float32)

# Combine features
def encode_CB(col1, col2):
    new_col_name = col1 + '_' + col2
    df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)

# Frequency encoding
def encode_FE(cols):
    for col in cols:
        vc = df[col].value_counts(normalize=True, dropna=True).to_dict()
        nm = col + '_FE'
        df[nm] = df[col].map(vc)
        df[nm] = df[nm].astype(np.float32)
        
# Label encoding
def encode_LE(cols):
    for col in cols:
        nm = col + '_LE'
        df[nm],_ = df[col].factorize(sort=True)
        df[nm] = df[nm].astype(np.float32)
        
def single_feat_cv(grps, c='FIELD_'):
    good_count = train.loc[train.label == 0, 'label'].count()
    bad_count = train.loc[train.label == 1, 'label'].count()
    
    for grp in grps:
        print('Cross Val for Group: ' + str(grp))
        for col in grp:
            col = c + str(col)
            if (df[col].apply(type) == str).any():
                cat_feat = [0]
            else:
                cat_feat = None
            catboost = CatBoostClassifier(iterations=1000,
                                      cat_features=cat_feat,
                                      class_weights= [1,good_count/bad_count],
                                      eval_metric='AUC', 
                                      colsample_bylevel=.8,
                                      verbose=0)
            cv = cross_validate(catboost, X=train[[col]], y=train['label'], scoring='roc_auc',
                                cv=5)

            print('*******')
            print('{} Avg.: {} \t Std.: {}'.format(col, np.mean(cv['test_score'] * 2 - 1), np.std(cv['test_score'] * 2 - 1)))
            print('*******')
        
def reduce_group(grps,c='FIELD_'):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = train[c+str(gg)].nunique()
            if n>mx:
                mx = n
                vx = gg
            
        use.append(vx)
        
    print('Use these',use)
nan_groups = {}
for col in train.columns:
    nan_counts = train[col].isnull().sum()
    try:
        nan_groups[nan_counts].append(col)
    except:
        nan_groups[nan_counts] = [col]

for nan_count, nan_group in nan_groups.items():
    print('#####')
    print(str(nan_group))
    print('NaN counts: ' + str(nan_count))
    print('#####')
with pd.option_context('display.max_columns', None):
    print(df.head())
    
# Count NaN/None
df_temp = df.loc[:, ~df.columns.str.contains('label')]
df['nan_count'] = df_temp.isnull().sum(axis=1)
df['none_count'] = df_temp.isin(['None', -1]).sum(axis=1)
df['age_match'] = df.loc[df.age_source1 == df.age_source2, 'age_source1']

# VN to EN
vn_to_eng_col('province')
vn_to_eng_col('maCv')
vn_to_eng_col('district')

# Null values
na_cols = ['province', 'district', 'maCv']
fixed_na(na_cols)

# age_source
df['age'] = df['age_source2'].fillna(df['age_source1'])
df.loc[df['age'] == -1, 'age'] = df['age_source1']
df['age_diff'] = df['age_source1'] - df['age_source2']

clean_field_7()
clean_field_9()
clean_field_11()
clean_field_12()

# # FIELD_13
# replace_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# for s in replace_str:
#     df['FIELD_13'].replace('^' + s + '.*', s, regex=True, inplace=True)
# df['FIELD_13'].replace('\d', np.nan, regex=True, inplace=True)
    
# FIELD_35
df['FIELD_35'].replace('Zero', 0, inplace=True)
df['FIELD_35'].replace('One', 1, inplace=True)
df['FIELD_35'].replace('Two', 2, inplace=True)
df['FIELD_35'].replace('Three', 3, inplace=True)
df['FIELD_35'].replace('Four', 4, inplace=True)
# df['FIELD_35'].fillna(-999, inplace=True)
df['FIELD_35'] = df['FIELD_35'].astype(np.float32)

# FIELD_39
df['FIELD_39'].replace('HQ', 'KR', inplace=True)
df['FIELD_39'].replace('TQ', 'CN', inplace=True)
df['FIELD_39'].replace('DL', 'TW', inplace=True)
df['FIELD_39'].replace('AN|AO|AT|CH|ID|WS|1', 'None', regex=True, inplace=True)
df['FIELD_39'].replace('AD|AE|BE|CA|GB|IT|N|SC|SE|TK|TL|TR|TS', np.nan, regex=True, inplace=True)

# FIELD_40
df['FIELD_40'].replace('05 08 11 02|02 05 08 11|08 02', 5, regex=True, inplace=True)
df['FIELD_40'].replace('None', -1, regex=True, inplace=True)
# df['FIELD_40'].fillna(-999, inplace=True)
df['FIELD_40'] = df['FIELD_40'].astype(np.float32)

# FIELD_41
df['FIELD_41'].replace('^I$', 1, regex=True, inplace=True)
df['FIELD_41'].replace('^II$', 2, regex=True, inplace=True)
df['FIELD_41'].replace('^III$', 3, regex=True, inplace=True)
df['FIELD_41'].replace('^IV$', 4, regex=True, inplace=True)
df['FIELD_41'].replace('^V$', 5, regex=True, inplace=True)
df['FIELD_41'].replace('None', -1, inplace=True)
# df['FIELD_41'].fillna(-999, inplace=True)
df['FIELD_41'] = df['FIELD_41'].astype(np.float32)

# FIELD_42
df['FIELD_42'].replace('Zezo', 0, inplace=True)
df['FIELD_42'].replace('One', 1, inplace=True)
df['FIELD_42'].replace('None', -1, inplace=True)
# df['FIELD_42'].fillna(-999, inplace=True)
df['FIELD_42'] = df['FIELD_42'].astype(np.float32)

# FIELD_43
df['FIELD_43'].replace('A', 1, inplace=True)
df['FIELD_43'].replace('C', 2, inplace=True)
df['FIELD_43'].replace('B', 3, inplace=True)
df['FIELD_43'].replace('D', 4, inplace=True)
df['FIELD_43'].replace('None', -1, inplace=True)
# df['FIELD_43'].fillna(-999, inplace=True)
df['FIELD_43'] = df['FIELD_43'].astype(np.float32)

# FIELD_44
df['FIELD_44'].replace('One', 1, inplace=True)
df['FIELD_44'].replace('Two', 2, inplace=True)
df['FIELD_44'].replace('None', -1, inplace=True)
# df['FIELD_44'].fillna(-999, inplace=True)
df['FIELD_44'] = df['FIELD_44'].astype(np.float32)    

# FIELD_45
df['FIELD_45'].replace('1', 1, inplace=True)
df['FIELD_45'].replace('2', 2, inplace=True)
df['FIELD_45'].replace('3', 3, inplace=True)
df['FIELD_45'].replace('None', -1, inplace=True)
# df['FIELD_45'].fillna(-999, inplace=True)
df['FIELD_45'] = df['FIELD_45'].astype(np.float32)

# Float to int
df['FIELD_51_int'] = df['FIELD_51'].fillna(-999).astype(np.int32)
df['FIELD_52_int'] = df['FIELD_52'].fillna(-999).astype(np.int32)
df['FIELD_53_int'] = df['FIELD_53'].fillna(-999).astype(np.int32)

# Replace True/False with 1/0
bool_cols = ['FIELD_18', 'FIELD_19', 'FIELD_20', 'FIELD_23', 'FIELD_25', 'FIELD_26',
             'FIELD_27', 'FIELD_28', 'FIELD_29', 'FIELD_30', 'FIELD_31', 'FIELD_36',
             'FIELD_37', 'FIELD_38', 'FIELD_47', 'FIELD_48', 'FIELD_49']
encode_BL(bool_cols, fix_na=False)

# Combine bool features
encode_CB('district', 'province')
encode_CB('FIELD_17', 'FIELD_24')
encode_CB('FIELD_7_0', 'FIELD_7_1')
encode_CB('FIELD_7_0_FIELD_7_1', 'FIELD_7_2')

# Frequency encoding
fe_cols = ['province', 'district', 'FIELD_9', 'FIELD_10', 
           'FIELD_17', 'FIELD_24', 'FIELD_17_FIELD_24',
           'FIELD_7_0', 'FIELD_7_1', 'FIELD_7_2', 'FIELD_7_3',
           'district_province', 'FIELD_7_0_FIELD_7_1', 'FIELD_7_0_FIELD_7_1_FIELD_7_2', 'FIELD_50', 'FIELD_54', 'FIELD_52', 'FIELD_56', 'FIELD_53', 'FIELD_57']
encode_FE(fe_cols)

# Label encoding
le_cols = ['FIELD_8', 'FIELD_11', 'FIELD_13', 'FIELD_17', 'FIELD_24', 'FIELD_17_FIELD_24']
encode_LE(le_cols)

# Nan encoding
# nan_cols = ['FIELD_2', 'FIELD_3', 'FIELD_4', 'FIELD_5', 'FIELD_6', 'FIELD_10_FE',
#             'FIELD_11_LE', 'FIELD_13_LE', 'FIELD_16', 'FIELD_21', 'FIELD_22', 'FIELD_23', 
#             'FIELD_25']
# encode_NAN(nan_cols)

df['FIELD_7_0_==_FIELD_9'] = np.where(df['FIELD_7_0'] == df['FIELD_9'], 1, 0)
df['FIELD_7_1_==_FIELD_9'] = np.where(df['FIELD_7_1'] == df['FIELD_9'], 1, 0)
df['FIELD_7_2_==_FIELD_9'] = np.where(df['FIELD_7_2'] == df['FIELD_9'], 1, 0)
df['FIELD_7_3_==_FIELD_9'] = np.where(df['FIELD_7_3'] == df['FIELD_9'], 1, 0)
with pd.option_context('display.max_rows', None):
    print(df.dtypes)
train = df.iloc[:30000, :]
test = df.iloc[30000:, :]
test.tail()
nan_groups = {}
for col in train.columns:
    nan_counts = train[col].isnull().sum()
    try:
        nan_groups[nan_counts].append(col)
    except:
        nan_groups[nan_counts] = [col]

for nan_count, nan_group in nan_groups.items():
    print('#####')
    print(str(nan_group))
    print('NaN counts: ' + str(nan_count))
    print('#####')
cols = [col for col in df.columns]
print(cols)
train.label.value_counts()
sns.countplot(train.label)
plt.title('Train label')
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape))
print('Total dataset shape: {}'.format(df.shape))
train.head().transpose()
with pd.option_context('display.max_rows', None):
    print(quick_analysis('province'))
train.groupby('province')['label'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)
df.loc[df.province == 'Tỉnh Đồng Nai', 'district'].value_counts()
with pd.option_context('display.max_rows', None):
    print(quick_analysis('district'))
quick_analysis('age_source1')
quick_analysis('age_source2')
train['age_source1'].describe()
train['age_source2'].describe()
df.loc[df['age_source2'] == -1, ['age_source1', 'label']]
df.loc[df['age_source1'] == 0, ['age_source2', 'label']]
with pd.option_context('display.max_rows', None):
    print(quick_analysis('maCv'))
vn_to_eng_col('maCv')
df.maCv.replace('^.*cnv.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*cn.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*c.n*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*c.nhan.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*cong nhan.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*coong nhaon.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*lao dong.*$', 'cong nhan', regex=True, inplace=True)
df.maCv.replace('^.*nhan vien.*$', 'nhan vien', regex=True, inplace=True)
df.maCv.replace('^.*can bo.*$', 'nhan vien', regex=True, inplace=True)
df.maCv.replace('^.*nv.*$', 'nhan vien', regex=True, inplace=True)
df.maCv.replace('^.*lai xe.*$', 'lai xe', regex=True, inplace=True)
df.maCv.replace('^.*tai xe.*$', 'lai xe', regex=True, inplace=True)
df.maCv.replace('^.*giao vien.*$', 'giao vien', regex=True, inplace=True)
df.maCv.replace('^.*gv.*$', 'giao vien', regex=True, inplace=True)
df.maCv.replace('^.*giang vien.*$', 'giao vien', regex=True, inplace=True)
# df.maCv.replace('(^.*[cn|nv|cnv|dao tao|boi keo|bao tri|bam gio|ho tro|tho].*$)', 'cong nhan vien', regex=True, inplace=True)

train = df.iloc[:30000, :]
test = df.iloc[30000:, :]
with pd.option_context('display.max_rows', None):
    print(quick_analysis('maCv'))
maCv_dict = {['cn', 'cnv']: 'cong nhan', 'tho phu': 'cong nhan', 'nv': 'nhan vien', 'cnv': 'cong nhan', }
train.loc[train.maCv.isnull() &
          ~train.age_source2.isnull(), 'maCv']
train.groupby('maCv')['label'].agg(['mean', 'sum', 'count']).sort_values('mean', ascending=False)
quick_analysis('FIELD_1')
quick_analysis('FIELD_2')
with pd.option_context('display.max_rows', None):
    print(quick_analysis('FIELD_3'))
quick_analysis('FIELD_4')
quick_analysis('FIELD_5')
quick_analysis('FIELD_6')
df.loc[df['FIELD_3'] > 8009, ['FIELD_4', 'FIELD_5', 'FIELD_6', 'label']]
quick_analysis('FIELD_7')
quick_analysis('FIELD_8')
quick_analysis('FIELD_9')
df.loc[df.FIELD_9 == 'BT', 'FIELD_7']
nan_groups[463].remove('FIELD_7')
df.FIELD_10.value_counts()
quick_analysis('FIELD_10')
train.loc[~train.FIELD_10.isnull() &
          train.maCv.isnull(), 'FIELD_10']
df.FIELD_11.value_counts().sort_index()
quick_analysis('FIELD_11')
(df.FIELD_11 / df.FIELD_22).value_counts()
df[['FIELD_11', 'FIELD_22', 'label']].corr()
quick_analysis('FIELD_12')
df.loc[df['FIELD_12'] == 'TN', 'FIELD_7']
df.loc[df['FIELD_12'] == 'XK', ['maCv', 'FIELD_7']]
df.loc[df['FIELD_11'] == '1', ['FIELD_7', 'FIELD_9', 'FIELD_13', 'FIELD_39', 'FIELD_41']]
with pd.option_context('display.max_rows', None):
    print(quick_analysis('FIELD_13'))
df['FIELD_13'].value_counts(sort=True).head(20)
df['FIELD_13'].value_counts(sort=True).tail(20)
df.loc[~df['FIELD_13'].isnull() & df['FIELD_13'].str.contains('^[0-9]', regex=True), ['FIELD_13']].value_counts()
df.FIELD_14.value_counts()
df.FIELD_15.value_counts()
df.loc[(df.FIELD_14 == 1) & (df.FIELD_15 == 0), 'label'].value_counts()
df.loc[df.FIELD_14 == df.FIELD_15, 'label'].value_counts()
df.loc[df.FIELD_14 != df.FIELD_15, 'label'].value_counts()
train[['FIELD_14', 'FIELD_15', 'label']].corr()
train.loc[train.province == 'missing', 'district'].value_counts()
quick_analysis('FIELD_16') 
quick_analysis('FIELD_17')
df.loc[~df['FIELD_17'].isnull() & df['FIELD_17'].str.contains('GX'), 'FIELD_24'].value_counts()
df.loc[~df['FIELD_17'].isnull() & df['FIELD_17'].str.contains('G2'), 'FIELD_24'].value_counts()
quick_analysis('FIELD_18')
quick_analysis('FIELD_19')
quick_analysis('FIELD_20')
quick_analysis('FIELD_21')
df.loc[df.FIELD_21 == 2, 'FIELD_16'].value_counts()
df.loc[df.FIELD_21 == 1, 'FIELD_16'].value_counts()

df.loc[df.FIELD_21 == 0, 'FIELD_16'].value_counts()
quick_analysis('FIELD_22')
quick_analysis('FIELD_23')
df.loc[df['FIELD_23'] == -999]
df.loc[df['FIELD_23'] == -999, 'FIELD_17_FIELD_24_LE'].value_counts()
df.loc[df['FIELD_23'].isnull(), 'FIELD_50'].value_counts()
quick_analysis('FIELD_24')
quick_analysis('FIELD_25')
quick_analysis('FIELD_26')
quick_analysis('FIELD_27')
quick_analysis('FIELD_28')
quick_analysis('FIELD_29')
quick_analysis('FIELD_12')
quick_analysis('FIELD_30')
quick_analysis('FIELD_31')
quick_analysis('FIELD_32')
quick_analysis('FIELD_33')
quick_analysis('FIELD_34')
quick_analysis('FIELD_35')
df.FIELD_35.value_counts()
quick_analysis('FIELD_36')
quick_analysis('FIELD_37')
quick_analysis('FIELD_38')
quick_analysis('FIELD_39')
edge_case = df.FIELD_39.value_counts().tail(33).index
edge_case = ['^' + ec + '$' for ec in edge_case]
edge_case = '|'.join(edge_case)
df.loc[~df.FIELD_39.isnull() & df.FIELD_39.str.contains(edge_case, regex=True), ['FIELD_39', 'label']].groupby('FIELD_39').mean()
df.loc[df['FIELD_39'] == '1', ['FIELD_7', 'FIELD_9']]
quick_analysis('FIELD_40')
quick_analysis('FIELD_41')
quick_analysis('FIELD_35')
df.loc[df['FIELD_41'] == 'V', 'FIELD_7'].head(20)
df.FIELD_42.value_counts()
quick_analysis('FIELD_43')
quick_analysis('FIELD_44')
quick_analysis('FIELD_45')
df.loc[df.FIELD_45 > 1, 'FIELD_11'].value_counts().sort_index()
quick_analysis('FIELD_11')
quick_analysis('FIELD_46')
quick_analysis('FIELD_47')
quick_analysis('FIELD_48')
quick_analysis('FIELD_49')
quick_analysis('FIELD_50')
quick_analysis('FIELD_51')
quick_analysis('FIELD_51_int')
quick_analysis('FIELD_24_LE')
with pd.option_context('display.max_rows', None):
    print(quick_analysis('FIELD_52'))
from decimal import *

with pd.option_context('display.max_rows', None):
    print(df.FIELD_52.apply(lambda x: x - np.int(x) if ~np.isnan(x) else np.nan).value_counts().sort_index())
with pd.option_context('display.max_rows', None):
    print(quick_analysis('FIELD_53'))
(df['FIELD_53_int'] - df['FIELD_52_int']).value_counts()
df[['FIELD_53_int', 'FIELD_52_int', 'label']].corr()
quick_analysis('FIELD_54')
quick_analysis('FIELD_55')
quick_analysis('FIELD_51')
quick_analysis('FIELD_56')
df.loc[~df.FIELD_52_FE.isnull() & (df.FIELD_52_FE != df.FIELD_56_FE), ['FIELD_52_FE', 'FIELD_56_FE']]
quick_analysis('FIELD_57')
nan_groups[0].remove('id')
nan_groups[0].remove('label')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[nan_groups[0]].corr(), ax=ax, annot=True)
grps=[[33],[34],[46],[47],['51_int','52_int','53_int'],[52,53,56,57],[50,54],[51,55],['17_LE','24_LE']]
reduce_group(grps)
grps=[['FIELD_32','nan_count']]
reduce_group(grps,c='')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[nan_groups[9678]].corr(), ax=ax, annot=True)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[nan_groups[9678]].corr(), ax=ax, annot=True)
grps=[[52,53,56,57],[50,54],[51,55]]
reduce_group(grps)
single_feat_cv([[11],[12, 29, 31],[16],[18],[19],[20],[21],[22],[25],[26],[27],[35],
                [36],[37],[38],[40],[41],[42],[43],[44,45],[50,54],[51,55],[52,53],[56,57]])
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[['FIELD_16', 'FIELD_21', 'FIELD_36']].corr(), ax=ax, annot=True)
single_feat_cv([['nan_count',
 'none_count',
 'age',
 'FIELD_7_count',
 'FIELD_51_int',
 'district_province',
 'FIELD_7_0_FIELD_7_1',
 'FIELD_7_0_FIELD_7_1_FIELD_7_2',
 'province_FE',
 'district_FE',
 'FIELD_9_FE',
 'FIELD_10_FE',
 'FIELD_7_0_FE',
 'FIELD_7_1_FE',
 'FIELD_7_2_FE',
 'FIELD_7_3_FE',
 'FIELD_8_LE',
 'FIELD_11_LE',
 'FIELD_13_LE',
 'FIELD_17_LE',
 'FIELD_24_LE',
 'FIELD_17_FIELD_24_LE',
 'FIELD_7_0_==_FIELD_9',
 'FIELD_7_1_==_FIELD_9',
 'FIELD_7_2_==_FIELD_9',
 'FIELD_7_3_==_FIELD_9']], c='')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[nan_groups[463]].corr(), ax=ax, annot=True)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df[['FIELD_1',
 'FIELD_4',
 'FIELD_5',
 'FIELD_6',
 'FIELD_12',
 'FIELD_15',
 'FIELD_16',
 'FIELD_18',
 'FIELD_19',
 'FIELD_20',
 'FIELD_21',
 'FIELD_22',
 'FIELD_25',
 'FIELD_26',
 'FIELD_28',
 'FIELD_30',
 'FIELD_32',
 'FIELD_33',
 'FIELD_34',
 'FIELD_35',
 'FIELD_36',
 'FIELD_37',
 'FIELD_38',
 'FIELD_39',
 'FIELD_40',
 'FIELD_41',
 'FIELD_42',
 'FIELD_43',
 'FIELD_44',
 'FIELD_45',
 'FIELD_46',
 'FIELD_47',
 'FIELD_48',
 'FIELD_49',
 'FIELD_50',
 'FIELD_52',
 'FIELD_55',
 'nan_count',
 'none_count',
 'age',
 'FIELD_7_count',
 'FIELD_51_int',
 'district_province',
 'FIELD_7_0_FIELD_7_1',
 'FIELD_7_0_FIELD_7_1_FIELD_7_2',
 'province_FE',
 'district_FE',
 'FIELD_9_FE',
 'FIELD_10_FE',
 'FIELD_7_0_FE',
 'FIELD_7_1_FE',
 'FIELD_7_2_FE',
 'FIELD_7_3_FE',
 'FIELD_8_LE',
 'FIELD_11_LE',
 'FIELD_13_LE',
 'FIELD_17_LE',
 'FIELD_24_LE',
 'FIELD_17_FIELD_24_LE',
 'FIELD_7_0_==_FIELD_9',
 'FIELD_7_1_==_FIELD_9',
 'FIELD_7_2_==_FIELD_9',
 'FIELD_7_3_==_FIELD_9']].corr(), ax=ax, annot=True)
single_feat_cv([[3],[4],[5],[6]])
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate

good_count = train.loc[train.label == 0, 'label'].count()
bad_count = train.loc[train.label == 1, 'label'].count()

train.loc[train['FIELD_24'].isnull(), 'FIELD_24'] = 'NaN'

catboost = CatBoostClassifier(iterations=1000,
                              cat_features=[0],
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_24']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))

catboost = CatBoostClassifier(iterations=1000,
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_24_FE']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))
# Best of 3 methods

catboost = CatBoostClassifier(iterations=1000,
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_24_LE']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))
train['FIELD_27'].fillna('NaN', inplace=True)
catboost = CatBoostClassifier(iterations=1000,
                              cat_features=[0],
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_17']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))
catboost = CatBoostClassifier(iterations=1000,
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_17_FE']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))
# Equal with FIELD_17_FE

catboost = CatBoostClassifier(iterations=1000,
                              class_weights= [1,good_count/bad_count],
                              eval_metric='AUC', 
                              colsample_bylevel=.8)
cv = cross_validate(catboost, X=train[['FIELD_17_LE']], y=train['label'], cv=5)
print(np.mean(cv['test_score'] * 2 - 1))
print(np.std(cv['test_score'] * 2 - 1))
with pd.option_context('display.max_rows', None):    
    print(df.dtypes)
