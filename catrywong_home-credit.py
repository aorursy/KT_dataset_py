#基本模块

import numpy as np

import pandas as pd

#画图模块

import matplotlib.pyplot as plt

import seaborn as sns

#模型训练前把数据分组用的train_test_split, 计算效果得分roc_auc_score, 用到基础模型xgboost,

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score 

import xgboost 

#忽略wainings

import warnings

warnings.filterwarnings('ignore')

#清理内存

import gc

#图表显示设置

%matplotlib inline

#打印数据表格的文件

import os

os.listdir("../input")
df_train = pd.read_csv('../input/application_train.csv')

df_test  = pd.read_csv('../input/application_test.csv')
df_train.head()
df_test.head()
df_all = pd.concat([df_train.loc[: , 'SK_ID_CURR':'AMT_REQ_CREDIT_BUREAU_YEAR'],

                   df_test.loc[: , 'SK_ID_CURR':'AMT_REQ_CREDIT_BUREAU_YEAR']])

df_all = df_all.reset_index(drop = True)

df_all.drop('TARGET', axis = 1, inplace = True)
print(df_train.shape, df_test.shape, df_all.shape)
#查找空缺值的列及其占比

def missing_values_table(df):

    miss_value = df.isnull().sum()

    miss_val_percent = 100 * df.isnull().sum() / len(df)

    miss_table = pd.concat([miss_value,miss_val_percent], axis = 1)

    miss_table = miss_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    miss_table = miss_table[miss_table.iloc[: , 1]!= 0].sort_values('% of Total Values', ascending=False).round(2)

    return miss_table
missing_values_table(df_train).head(10)
missing_values_table(df_test).head(10)
distribution_of_target = df_train['TARGET'].value_counts()

print(distribution_of_target)
sns.countplot(x = 'TARGET',data = df_train)

perc_target = (100 * df_train['TARGET'].sum() / df_train['TARGET'].count()).round(4)

print('percentage of default : %0.2f%%' %  perc_target)
#区分开文本列和数字列

feat_obj = df_all.dtypes[df_all.dtypes == 'object'].index

feat_num = df_all.dtypes[df_all.dtypes != 'object'].index
#看看文本列有多少不同的值（可用train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)）

df_all[feat_obj].apply(pd.Series.nunique, axis = 0)
df_all['CODE_GENDER'].value_counts()
df_all['CODE_GENDER'].replace('XNA', np.nan, inplace = True )
feat_obj_dum = pd.get_dummies(df_all[feat_obj], dummy_na = True)

df_all = pd.concat([df_all, feat_obj_dum], axis = 1)

#删除原文本列

df_all.drop(feat_obj,axis = 1, inplace = True)
feat_obj_dum['TARGET'] = df_train['TARGET']
obj_corr = feat_obj_dum.corr()

obj_corr = obj_corr['TARGET']
abs(obj_corr).sort_values(ascending = False).head(10)
del feat_obj_dum

gc.collect()
df_all.loc[ : , ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']].describe()
df_all['DAYS_EMPLOYED'].plot.hist()

plt.xlabel('DAYS_EMPLOYED')
anom = df_train[df_train['DAYS_EMPLOYED'] == 365243]

nom = df_train[df_train['DAYS_EMPLOYED'] != 365243]

prec_anom = 100 * anom['TARGET'].mean()

prec_nom = 100 * nom['TARGET'].mean()



print('number of anomalies :', len(anom))

print('percent of anomalies that default the loans : %0.2f%%' % prec_anom)

print('percent of nomalies that default the loans :  %0.2f%%' % prec_nom)
del anom, nom

gc.collect()
df_all['DAYS_EMPLOYED_anom'] = df_all['DAYS_EMPLOYED'] == 365243

df_all['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
anom_dum = pd.get_dummies(df_all['DAYS_EMPLOYED_anom'], dummy_na = True)

df_all = pd.concat([df_all, anom_dum], axis = 1)

df_all.drop(['DAYS_EMPLOYED_anom'],axis = 1, inplace = True)
del anom_dum

gc.collect()
#工作时间占年龄的比率

df_all['DAY_EMPLOYED_PERC'] = df_all['DAYS_EMPLOYED'] / df_all['DAYS_BIRTH']

#总收入占贷款的比率

df_all['INCOME_CREDIT_PERC'] = df_all['AMT_INCOME_TOTAL'] / df_all['AMT_CREDIT']

#该用户家庭的人均收入

df_all['INCOME_PER_PERSON'] = df_all['AMT_INCOME_TOTAL'] / df_all['CNT_FAM_MEMBERS']

#贷款年金占总收入的比率

df_all['ANNUITY_INCOME_PERC'] = df_all['AMT_ANNUITY'] / df_all['AMT_INCOME_TOTAL']

#贷款年金占贷款的比率

df_all['PAYMENT_RATE'] = df_all['AMT_ANNUITY'] / df_all['AMT_CREDIT']
#用train表数据，outliers表示需要删除的行索引

outlier_indices = []

for i in feat_num:

    Q1 = df_train[i].quantile(0.02)

    Q3 = df_train[i].quantile(0.98)

    IQR = Q3 - Q1

    outliers = df_train[(df_train[i] < Q1 - 1.5 * IQR) | (df_train[i] > Q3 + 1.5 * IQR)].index

    outlier_indices.extend(outliers)
#可能存在重复的行索引，先去重

from collections import Counter

outlier_indices = Counter(outlier_indices)#字典形式出现

multiple_outliers = []

for key, values in outlier_indices.items():#字典.items() 函数以列表返回可遍历的(键, 值) 元组数组

    if values > 2:

        multiple_outliers.append(key)  
#需删掉的行数

len(multiple_outliers)

df_all.drop(multiple_outliers, inplace = True)
#train数据集去除outliers后的行列数

df_all.shape
tem = df_train[feat_num]

tem["TARGET"] = df_train["TARGET"]

num_corr = tem.corr()
abs(num_corr['TARGET']).sort_values(ascending = False)
num_value_count = df_all[feat_num].apply(pd.Series.nunique, axis = 0)
feat_num_dum = num_value_count[num_value_count <= 150].index.tolist()

feat_num_not_dum = num_value_count[num_value_count > 150].index.tolist()

print('There are %d feature of num need to get dummy.' % len(feat_num_dum))

print('There are %d feature of num left.' % len(feat_num_not_dum))
df_all = pd.get_dummies(df_all, columns = feat_num_dum, dummy_na = True )
df_all.shape
df_train['EXT_SOURCE_1'].value_counts().sort_values(ascending = False)
ext_source = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

df_ext_source = df_all[ext_source]

for i in ext_source:

    print(df_ext_source[i].isnull().sum())

    

df_ext_source.dropna(inplace = True)

print(df_ext_source.head(20))
plt.figure(figsize = (15, 15))

for i, col in enumerate(ext_source):

    plt.subplot(3, 1, i + 1)

    plt.hist(df_ext_source[col], bins = 5000, color = 'blue')

    plt.title('distribution of %s' % col)

    plt.xlabel('%s' % col)   
#新增一列EXT_SOURCE_1_null，表示EXT_SOURCE_1是否为空值，空值记为1

df_all['EXT_SOURCE_1_null'] = 0

df_all['EXT_SOURCE_1_null'][df_all['EXT_SOURCE_1'].isnull()] = 1
#验证是否有遗漏

df_all['EXT_SOURCE_1_null'].sum()
#新增一列，表示是否为空值。空值记为1

df_all['EXT_SOURCE_2_null'] = 0

df_all['EXT_SOURCE_2_null'][df_all['EXT_SOURCE_2'].isnull()] = 1
#验证是否有遗漏

df_all['EXT_SOURCE_2_null'].sum()
#查看计数量较多的值

tem = df_all['EXT_SOURCE_2'].value_counts().sort_values(ascending = False)
#计数量大于100的值有21个

tem[tem > 100].shape[0]
#计数量大于100的值生成dummy列

for i in tem[tem > 100].index:

    df_all['EXT_SOURCE_2' + str(i)] = 0

    df_all['EXT_SOURCE_2' + str(i)][df_all['EXT_SOURCE_2'] == i] = 1
df_all.shape
#新增一列，表示是否为空值。空值记为1

df_all['EXT_SOURCE_3_null'] = 0

df_all['EXT_SOURCE_3_null'][df_all['EXT_SOURCE_3'].isnull()] = 1
#验证是否有遗漏

df_all['EXT_SOURCE_3_null'].sum()
#查看计数量较多的值

tem = df_all['EXT_SOURCE_3'].value_counts().sort_values(ascending = False)
#计数量大于1000的值有48个

tem[tem > 1000].shape[0]
#计数量大于1000的值生成dummy列

for i in tem[tem > 1000].index:

    df_all['EXT_SOURCE_3' + str(i)] = 0

    df_all['EXT_SOURCE_3' + str(i)][df_all['EXT_SOURCE_3'] == i] = 1
df_all.shape
df_all.fillna(-1, inplace = True)
#检验是有全部填好空值

df_all.isnull().sum()
bureau = pd.read_csv('../input/bureau.csv')

bb = pd.read_csv('../input/bureau_balance.csv')
#定义对dataframe做one hot的函数。其中只对‘object’类型的列做转化。

def one_hot_encoder(df, nan_category = True):

    original_cols = list(df.columns)

    categorial_cols = [col for col in df.columns if df[col].dtypes == 'object']

    df = pd.get_dummies(df, columns = categorial_cols, dummy_na = nan_category)

    new_columns = [i for i in df.columns if i not in original_cols]

    return df, new_columns
bureau.head()
bb.head()
b_obj = bureau.dtypes[bureau.dtypes == 'object'].index

bb_obj = bb.dtypes[bb.dtypes == 'object'].index
bureau[b_obj].apply(pd.Series.nunique, axis = 0)
bb[bb_obj].apply(pd.Series.nunique, axis = 0)
bureau, bureau_cat = one_hot_encoder(bureau)

bb, bb_cat = one_hot_encoder(bb)
bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size'] } #统计最小值，最大值，出现笔数

for i in bb_cat:

    bb_aggregations[i] = ['mean','sum'] #算出占比和总次数



bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

tem = []

for i in bb_agg.columns.tolist():

    tem.append(i[0] + '_' + i[1])

bb_agg.columns = pd.Index(tem)
bb_agg.head()
#按SK_ID_BUREAU连接bureau和bb表

bureau = bureau.join(bb_agg, how= 'left', on='SK_ID_BUREAU')

bureau.drop(['SK_ID_BUREAU'], axis= 1, inplace= True)
del bb_agg, bb

gc.collect()
bureau.head()
#最后统计每个SK_ID_CURR的历史SK_ID_BUREAU信息。对所有数值计算最小值，最大值，均值，方差，总和

tem = bureau.columns.tolist()

num_agg = { }

for i in tem:

    if i != 'SK_ID_CURR':

        num_agg[i] = ['min','max','mean','var','sum']

bureau_agg = bureau.groupby('SK_ID_CURR', as_index = False).agg(num_agg)



tem = [ ]

tem.append('SK_ID_CURR')

for i in bureau_agg.columns.tolist():

    if i[0] != 'SK_ID_CURR':

        tem.append('bureau' + '_' + i[0] + '_' + i[1])

bureau_agg.columns = pd.Index(tem)
bureau_agg.head()
#拼接到df_all表上，记为df_allX

df_allX = df_all.merge(bureau_agg, how = 'left', on= 'SK_ID_CURR')
df_allX.shape
del bureau, bureau_agg

gc.collect()
ip = pd.read_csv('../input/installments_payments.csv')
ip.head()
#新增一列DEFAULT_DAY，表示每笔的逾期天数

ip['DEFAULT_DAY'] = ip['DAYS_ENTRY_PAYMENT'] - ip['DAYS_INSTALMENT']

tem = ip[ip['DEFAULT_DAY'] > 0]#逾期天数大于0

tem = tem.loc[: ,['SK_ID_CURR','DEFAULT_DAY']]

#计算每个SK_ID_CURR的逾期总天数和逾期总次数

default_days_agg = tem.groupby('SK_ID_CURR', as_index = False).agg({ 'DEFAULT_DAY' : [ 'sum', 'count']})



print(default_days_agg.head())
tem = []

tem.append('SK_ID_CURR')

for i in default_days_agg.columns.tolist():

    if i[0] != 'SK_ID_CURR':

        tem.append('ip' + '_' + i[0] + '_' + i[1])

default_days_agg.columns = pd.Index(tem)



print(default_days_agg.head())
df_allX = df_allX.merge(default_days_agg, how = 'left', on = 'SK_ID_CURR')
del default_days_agg

gc.collect()
ip['MONEY'] = ip['AMT_INSTALMENT'] - ip['AMT_PAYMENT']

tem = ip.loc[: ,['SK_ID_PREV', 'SK_ID_CURR', 'MONEY']]
tem = tem.groupby(['SK_ID_CURR','SK_ID_PREV'], as_index = False).count().rename(columns={'MONEY': 'TIMES'})
tem = tem.groupby('SK_ID_CURR', as_index = False).agg({'SK_ID_PREV': 'count', 'TIMES': 'sum'})
tem = tem.rename(index = str, columns = {"SK_ID_PREV": "ip_CREDIT_TIMES", "TIMES": "ip_TOTAL_INSTALLMENT_TIMES"})
tem.head()
df_allX = df_allX.merge(tem,how = 'left', on = 'SK_ID_CURR')
df_allX.shape
tem = ip.groupby('SK_ID_CURR', as_index = False).agg({'AMT_INSTALMENT':'sum', 'AMT_PAYMENT':'sum'})

tem['ip_default_money'] = tem['AMT_INSTALMENT'] - tem['AMT_PAYMENT']

tem = tem.loc[:,['SK_ID_CURR','ip_default_money']]
df_allX = df_allX.merge(tem, how = 'left', on= 'SK_ID_CURR')
del ip, tem

gc.collect()
pa = pd.read_csv('../input/previous_application.csv')
pa.head()
pa, pa_cat = one_hot_encoder(pa)
#新加一列特征：申请金额占与实际发放金额的比值

pa['application_credit_perc'] = pa['AMT_APPLICATION']/ pa['AMT_CREDIT']
cat_agg = { }

for i in pa_cat:

    cat_agg[i] = ['mean', 'sum']

    

num_agg = {

    'AMT_ANNUITY': ['min','max','mean'],

    'AMT_APPLICATION':['min','max','mean'],

    'AMT_CREDIT':['min','max','mean'],

    'AMT_DOWN_PAYMENT':['min','max','mean'],

    'AMT_GOODS_PRICE':['min','max','mean'],

    'application_credit_perc':['min','max','mean','var'],

    'HOUR_APPR_PROCESS_START':['min','max','mean'],

    'RATE_DOWN_PAYMENT':['min','max','mean'],

    'DAYS_DECISION':['min','max','mean'],

    'CNT_PAYMENT':['sum','mean']

}



pa_agg = pa.groupby('SK_ID_CURR', as_index = False).agg({**cat_agg, **num_agg})
pa_agg.head()
tem = []

tem.append('SK_ID_CURR')

for i in pa_agg.columns.tolist():

    if i[0] != 'SK_ID_CURR':

        tem.append(i[0] + '_' + i[1])

pa_agg.columns = pd.Index(tem)
df_allX = df_allX.merge(pa_agg, how = 'left', on = 'SK_ID_CURR')
df_allX.head()
del pa_agg, pa

gc.collect()
pcb = pd.read_csv('../input/POS_CASH_balance.csv')
pcb['STATUS_COMPLETED'] = 0

for i in range(pcb['NAME_CONTRACT_STATUS'].shape[0]):

    if pcb['NAME_CONTRACT_STATUS'].values[i] == 'Completed':

        pcb['STATUS_COMPLETED'].values[i] = 1
pcb, pcb_cat = one_hot_encoder(pcb)
pcb_cat_agg = { }

for i in pcb_cat:

    pcb_cat_agg[i] = ['mean','sum']



pcb_num_agg = { 

    'MONTHS_BALANCE': ['max', 'mean', 'size'],

    'SK_DPD': ['max', 'mean'],

    'SK_DPD_DEF': ['max', 'mean'],

    'STATUS_COMPLETED':['sum']

}
pcb_agg = pcb.groupby('SK_ID_CURR', as_index = False).agg({**pcb_cat_agg, **pcb_num_agg})
tem = [ ]

tem.append('SK_ID_CURR')

for i in pcb_agg.columns.tolist():

    if i[0] != 'SK_ID_CURR':

        tem.append('pcb' + '_' + i[0] + '_' + i[1])

pcb_agg.columns = pd.Index(tem)
#新增一列:每个用户的记录数

pcb_agg['pcb_count'] = pcb.groupby('SK_ID_CURR').size()
pcb_agg.head()
pcb.head()
total_completed = pcb.loc[ : , ['SK_ID_CURR', 'STATUS_COMPLETED']].groupby('SK_ID_CURR', as_index = False).sum()

credt_count = pcb.loc[ : , ['SK_ID_CURR', 'SK_ID_PREV']].groupby(

    'SK_ID_CURR', as_index = False).count().rename(columns = {'SK_ID_PREV' : 'pcb_PREV_CREDIT_COUNT'})

tem = total_completed.merge(credt_count, how = 'left', on = 'SK_ID_CURR')

tem['pcb_COMPLETED_PERC'] = tem['STATUS_COMPLETED'] / tem['pcb_PREV_CREDIT_COUNT']

print(tem.head())
pcb_agg = pcb_agg.merge(tem, how = 'left', on = 'SK_ID_CURR')
pcb_agg.head()
df_allX = df_allX.merge(pcb_agg, how = 'left', on= 'SK_ID_CURR')
df_allX.head()
del pcb, pcb_agg

gc.collect()
ccb = pd.read_csv('../input/credit_card_balance.csv')
ccb['STATUS_COMPLETED'] = 0

for i in range(ccb['NAME_CONTRACT_STATUS'].shape[0]):

    if ccb['NAME_CONTRACT_STATUS'].values[i] == 'Completed':

        ccb['STATUS_COMPLETED'].values[i] = 1
ccb, ccb_cat = one_hot_encoder(ccb)
ccb_cat_agg = { }

for i in ccb_cat:

    ccb_cat_agg[i] = ['mean','sum']

    

ccb_num_agg = { 

    'MONTHS_BALANCE': ['max', 'mean', 'size'],

    'SK_DPD': ['max', 'mean'],

    'SK_DPD_DEF': ['max', 'mean'],

    'STATUS_COMPLETED':['sum']

}



ccb_num = ['AMT_BALANCE',

       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',

       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',

       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',

       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',

       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',

       'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',

       'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',

       'CNT_INSTALMENT_MATURE_CUM']

for i in ccb_num:

    ccb_num_agg[i] = ['min', 'max', 'mean', 'sum', 'var']
ccb_agg = ccb.groupby('SK_ID_CURR', as_index = False).agg({**ccb_cat_agg, **ccb_num_agg})
tem = [ ]

tem.append('SK_ID_CURR')

for i in ccb_agg.columns.tolist():

    if i[0] != 'SK_ID_CURR':

        tem.append('ccb' + '_' + i[0] + '_' + i[1])

ccb_agg.columns = pd.Index(tem)
ccb_agg['ccb_count'] = ccb.groupby('SK_ID_CURR').size()
total_completed = ccb.loc[ : , ['SK_ID_CURR', 'STATUS_COMPLETED']].groupby('SK_ID_CURR', as_index = False).sum()

credt_count = ccb.loc[ : , ['SK_ID_CURR', 'SK_ID_PREV']].groupby(

    'SK_ID_CURR', as_index = False).count().rename(columns = {'SK_ID_PREV' : 'ccb_PREV_CREDIT_COUNT'})

tem = total_completed.merge(credt_count, how = 'left', on = 'SK_ID_CURR')

tem['ccb_COMPLETED_PERC'] = tem['STATUS_COMPLETED'] / tem['ccb_PREV_CREDIT_COUNT']

print(tem.head())
ccb_agg = ccb_agg.merge(tem, how = 'left',on = 'SK_ID_CURR')
ccb_agg.head()
df_allX = df_allX.merge(ccb_agg,how = 'left', on= 'SK_ID_CURR')
df_allX.head()
del ccb, ccb_agg, total_completed, credt_count ,tem

gc.collect()
df_allX.fillna(0, inplace = True)
#取出去掉outliers行后的target

df_target_fin = df_train.drop(multiple_outliers)[['TARGET']].copy()
print(df_target_fin.shape)



del df_train

gc.collect()
df_allX.drop('SK_ID_CURR',axis = 1, inplace = True )

df_train_fin = df_allX.loc[ : df_target_fin.shape[0]-1, : ].copy()

df_test_fin = df_allX.loc[df_target_fin.shape[0] : , : ].copy()

print(df_train_fin.shape,df_test_fin.shape)



del df_allX

gc.collect()