# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root = "/kaggle/input/unsw-nb15/"

train = pd.read_csv(root+"UNSW_NB15_training-set.csv")

test = pd.read_csv(root+"UNSW_NB15_testing-set.csv")

list_events = pd.read_csv(root+"UNSW-NB15_LIST_EVENTS.csv")

features = pd.read_csv(root+"NUSW-NB15_features.csv", encoding='cp1252')
print(train.shape, test.shape)

if train.shape[0]<100000:

    print("Train test sets are reversed. Fixing them.")

    train, test = test, train
train['type'] = 'train'

test['type'] ='test'

total = pd.concat([train, test], axis=0, ignore_index=True)

total.drop(['id'], axis=1, inplace=True)

# del train, test
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('object')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def standardize(df):

    return (df-df.mean())/df.std()

    

def min_max(df):

    return (df-df.min())/(df.max() - df.min())



def normalize(df):

    return pd.Dataframe(preprocessing.normalize(df), columns=df.columns)
total = reduce_mem_usage(total)
list_events.shape
list_events.head()
list_events['Attack category'].unique()
list_events['Attack subcategory'].unique()
features.head(features.shape[0])
# the Name column has camel case values

features['Name'] = features['Name'].str.lower()

# the following 4 columns are address related and not in train dataset

features = features[~features['Name'].isin(['srcip', 'sport', 'dstip', 'dsport'])].reset_index()

features.drop(['index', 'No.'], axis=1, inplace=True)
normal = train[train['label']==0]

anomaly = train[train['label']==1]
print(sorted(set(train.columns) - set(features['Name'].values)))

print(sorted(set(features['Name'].values) - set(train.columns)))
fix = {'ct_src_ ltm': 'ct_src_ltm', 'dintpkt': 'dinpkt', 'dmeansz': 'dmean', 'res_bdy_len': 'response_body_len', 'sintpkt': 'sinpkt', 'smeansz': 'smean'}

features['Name'] = features['Name'].apply(lambda x: fix[x] if x in fix else x)

features.to_csv('features.csv')
print(sorted(set(train.columns) - set(features['Name'].values)))

print(sorted(set(features['Name'].values) - set(train.columns)))
train.head()
train.dtypes
def show_correlation(data, method='pearson'):

    correlation_matrix = data.corr(method='pearson') #  ‘pearson’, ‘kendall’, ‘spearman’

    fig = plt.figure(figsize=(12,9))

    sns.heatmap(correlation_matrix,vmax=0.8,square = True) #  annot=True, if fig should show the correlation score too

    plt.show()

    return correlation_matrix



def top_correlations(correlations, limit=0.9):

    columns = correlations.columns

    for i in range(correlations.shape[0]):

        for j in range(i+1, correlations.shape[0]):

            if correlations.iloc[i,j] >= limit:

                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")

def print_correlations(correlations, col1=None, col2=None):

    columns = correlations.columns

    for i in range(correlations.shape[0]):

        for j in range(i+1, correlations.shape[0]):

            if (col1 == None or col1==columns[i]) and (col2 == None or col2==columns[j]):

                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")

                return

            elif (col1 == None or col1==columns[j]) and (col2 == None or col2==columns[i]):

                print(f"{columns[i]} {columns[j]} {correlations.iloc[i,j]}")

                return

            

def find_corr(df1, df2):

    return pd.concat([df1, df2], axis=1).corr().iloc[0,1]



def corr(col1, col2='label', df=total):

    return pd.concat([df[col1], df[col2]], axis=1).corr().iloc[0,1]
correlation_matrix = show_correlation(total)
top_correlations(correlation_matrix, limit=0.9)
correlation_matrix = show_correlation(train, method='spearman')
top_correlations(correlation_matrix, limit=0.9)
sns.pairplot(total[['spkts', 'sbytes', 'sloss']])
sns.pairplot(total[['dpkts', 'dbytes', 'dloss']])
sns.pairplot(total[['sinpkt', 'is_sm_ips_ports']])
sns.pairplot(total[['swin', 'dwin']])
def dual_plot(col, data1=normal, data2=anomaly, label1='normal', label2='anomaly', method=None):

    if method != None:

        sns.distplot(data1[col].apply(method), label=label1, hist=False, rug=True)

        sns.distplot(data2[col].apply(method), label=label2, hist=False, rug=True)

    else:

        sns.distplot(data1[col], label=label1, hist=False, rug=True)

        sns.distplot(data2[col], label=label2, hist=False, rug=True)

    plt.legend()

    

def catplot(data, col):

    ax = sns.catplot(x=col, hue="label", col="type",data=data, kind="count", height=5, legend=False, aspect=1.4)

    ax.set_titles("{col_name}")

    ax.add_legend(loc='upper right',labels=['normal','attack'])

    plt.show(ax)
def create_count_df(col, data=total):

    df = pd.DataFrame(data[col].value_counts().reset_index().values, columns = [col, 'count'])

    df['percent'] = df['count'].values*100/data.shape[0]

    return df.sort_values(by='percent', ascending=False)
create_count_df('label', train)
create_count_df('label', test)
col = 'state'

create_count_df(col, train)
# all other values those were few in train set, have been renamed to 'RST_and_others'

total.loc[~total[col].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), col] = 'others'

catplot(total, col)

# catplot(total[~total[col].isin(['INT', 'FIN', 'REQ', 'CON'])], col)
col = 'service'

create_count_df(col, train)
catplot(total[~total[col].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3'])], col)
total.loc[~total[col].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), col] = 'others'
col = 'proto'

create_count_df(col, normal)
create_count_df(col, anomaly)[:10]
# icmp and rtp columns are in test, but not in train data

total.loc[total[col].isin(['igmp', 'icmp', 'rtp']), col] = 'igmp_icmp_rtp'

total.loc[~total[col].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), col] = 'others'
catplot(total, 'is_sm_ips_ports')
col = 'is_ftp_login'

print(corr('ct_ftp_cmd', col), corr('is_ftp_login', 'label'))

catplot(total, col)

total.drop([col], axis=1, inplace=True)
col = 'ct_state_ttl'

catplot(total, col)
catplot(total, 'ct_ftp_cmd')

corr('ct_ftp_cmd', 'label')
col = 'ct_flw_http_mthd'

catplot(total, col)

corr(col) # -0.012237160723
create_count_df(col, total)
print(find_corr(total['spkts']*total['smean'], total['sbytes'])) # 0.999999

print(find_corr(total['dpkts']*total['dmean'], total['dbytes'])) # 0.99999

print(corr('sbytes', 'sloss'), corr('dbytes', 'dloss')) # 0.995771577240429, 0.9967111338305503

total.drop(['sbytes', 'dbytes'], axis=1, inplace=True)
dual_plot('smean')
dual_plot('dmean')
total['smean_log1p'] = total['smean'].apply(np.log1p)

total['dmean_log1p'] = total['dmean'].apply(np.log1p)



# -0.02837244879012871 -0.2951728296856902 -0.05807468815031313 -0.5111549621216057

print(corr('smean'), corr('dmean'), corr('smean_log1p'), corr('dmean_log1p'))

# So we have better correlation with label after applying log1p. 

total.drop(['smean', 'dmean'], axis=1, inplace=True)
col = 'spkts'

dual_plot(col)
dual_plot(col, method=np.log1p)
total['spkts_log1p'] = total['spkts'].apply(np.log1p)

total['dpkts_log1p'] = total['dpkts'].apply(np.log1p)



# -0.043040466783819634 -0.09739388286233619 -0.3468819761209388 -0.45005074723539357

print(corr('spkts'), corr('dpkts'), corr('spkts_log1p'), corr('dpkts_log1p'))

# So we have better correlation with label after applying log1p. 

total.drop(['spkts', 'dpkts'], axis=1, inplace=True)
col = 'sttl'

dual_plot(col) # 0.62408238, after applying log1p 0.61556952425
col = 'dttl'

dual_plot(col) # corr -0.09859087338578788
dual_plot('sloss')
# So log1p makes it easier to differentiate

dual_plot('sloss', method=np.log1p)
total['sloss_log1p'] = total['sloss'].apply(np.log1p)

total['dloss_log1p'] = total['dloss'].apply(np.log1p)

# 0.001828274080103508 -0.07596097807462938 -0.3454351103223904 -0.3701913238787703

print(corr('sloss'), corr('dloss'), corr('sloss_log1p'), corr('dloss_log1p') )

total.drop(['sloss', 'dloss'], axis=1, inplace= True)
total['swin'].value_counts().loc[lambda x: x>1]
total['dwin'].value_counts().loc[lambda x: x>1]
print(corr('swin'), corr('dwin'))
dual_plot('swin')
selected = ['swin', 'dwin']

kbins = preprocessing.KBinsDiscretizer(n_bins=[3, 3], encode='ordinal', strategy='uniform')

total[selected] = pd.DataFrame(kbins.fit_transform(total[selected]), columns=selected)

print(corr('swin'), corr('dwin'))
col = 'stcpb'

dual_plot(col)
dual_plot(col, method=np.log1p)
total['stcpb_log1p'] = total['stcpb'].apply(np.log1p)

total['dtcpb_log1p'] = total['dtcpb'].apply(np.log1p)

# -0.2665849100492664 -0.2635428109654134 -0.33898970769021913 -0.33835676091281974

print(corr('stcpb'), corr('dtcpb'), corr('stcpb_log1p'), corr('dtcpb_log1p'))

total.drop(['stcpb', 'dtcpb'], axis=1, inplace= True)
total.drop(['tcprtt'], axis=1, inplace=True)
dual_plot('synack')
dual_plot('ackdat')
col = 'trans_depth'

print(corr(col)) # -0.0022256544

create_count_df(col, total)
col = 'response_body_len'

dual_plot(col)
total["response_body_len_log1p"] = total["response_body_len"].apply(np.log1p)



# slight improve

# -0.018930127454048158 -0.03261972203078345

print(corr('response_body_len'), corr('response_body_len_log1p'))

total.drop(['response_body_len'], axis=1, inplace=True)
col = 'ct_srv_src'

print(total[col].value_counts())
print(corr(col)) # 0.24659616767

dual_plot(col)
col = 'ct_srv_dst'

print(total[col].value_counts())

# graph is same as ct_srv_src

dual_plot(col)
# 0.2478122357. they are very correlated 0.97946681, need to check whether dropping one benefits

print(corr('ct_srv_dst'), corr('ct_srv_src', 'ct_srv_dst'))
col = 'ct_src_ltm'

print(corr(col))

create_count_df(col, total)
print(corr('ct_dst_ltm'))

create_count_df('ct_dst_ltm', total)
corr('ct_src_ltm', 'ct_dst_ltm')
for col in ['ct_src_dport_ltm', 'ct_dst_sport_ltm']:

    print(corr(col))

    print(create_count_df(col, total))
corr('ct_src_dport_ltm', 'ct_dst_sport_ltm')
col = 'dur'

print(corr(col)) # 0.0290961170, correlation gets worse after log1p

dual_plot(col)
col = 'rate'

print(corr(col))

dual_plot(col) # cor 0.3358, after applying log1p it becomes 0.31581108
col = 'sinpkt'

corr(col, 'is_sm_ips_ports')
print(corr(col)) # corr -0.1554536980863

dual_plot(col) 
dual_plot(col, method=np.log1p)
dual_plot('dinpkt')
total['sinpkt_log1p'] = total['sinpkt'].apply(np.log1p)

total['dinpkt_log1p'] = total['dinpkt'].apply(np.log1p)



# slight improve in correlation

# -0.1554536980867726 -0.030136042428744566 -0.16119699304378052 -0.07408113676641241

print(corr('sinpkt'), corr('dinpkt'), corr('sinpkt_log1p'), corr('dinpkt_log1p'))

total.drop(['sinpkt', 'dinpkt'], axis=1, inplace= True)
dual_plot('sload')
dual_plot('dload')
total['sload_log1p'] = total['sload'].apply(np.log1p)

total['dload_log1p'] = total['dload'].apply(np.log1p)

# 0.16524867685764016 -0.35216880416636837 0.3397788822586144 -0.5919440288535992

print(corr('sload'), corr('dload'), corr('sload_log1p'), corr('dload_log1p'))

total.drop(['sload', 'dload'], axis=1, inplace=True)
dual_plot('sjit')
dual_plot('djit')
features.to_csv('features.csv', index=False)

train = total[total['type']=='train'].drop(['type'], axis=1)

test = total[total['type']!='train'].drop(['type'], axis=1)

train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)