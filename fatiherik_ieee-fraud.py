# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# train_trans=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
# train_trans
# target=train_trans[['TransactionID','isFraud']]
# target
# test_trans=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
# test_trans
train_iden=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_iden
test_iden=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_iden
test_iden.columns=list(train_iden.columns)
test_iden
train_iden=pd.concat([train_iden,test_iden])
train_iden
train_iden.columns
pd.set_option('display.max_columns', None)
cols=train_iden.columns.tolist()
cols
train_iden.isnull().sum()
(np.sum(pd.isnull(train_iden)).sort_values(ascending=False)/len(train_iden))*100
train_iden.info()
for i in [1,2,3,4,5,6,7,8,9,10,11,13,14,17,18,19,20,21,22,24,25,26,32]:
    
    if i<10:
        x='id_0'+str(i)
    else:
        x='id_'+str(i)
        
    if train_iden[x].nunique()>100:
        num_of_unique=100
    else:
        num_of_unique=train_iden[x].nunique()
    
    print('Column: '+x)
    print('--------------------------------------')
    print('Number of unique value: '+ str(train_iden[x].nunique()))
    print('--------------------------------------')
    print(train_iden[x].value_counts(dropna=False, normalize=True))
    
    plt.hist(train_iden[x], bins=num_of_unique)
    plt.title('Distribution of '+x+' variable')
    plt.show()
#print(train_iden['id_01'].nunique())
#print(train_iden['id_01'].value_counts(dropna=False, normalize=True))

#plt.hist(train_iden['id_01'], bins=train_iden['id_01'].nunique());
#plt.title('Distribution of id_01 variable');
for i in ['id_12', 'id_15', 'id_16', 'id_23','id_27','id_28', 'id_29', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']:
    df=train_iden[i].fillna('Null').value_counts(dropna=False).reset_index().rename(columns={'index':'title',i:'value'})
    sns.barplot(x='title', y='value', data=df).set_title(i)
    plt.show()
train_iden.id_30.value_counts(dropna=False, normalize=True)
train_iden.id_31.value_counts(dropna=False, normalize=True)
train_iden.id_33.value_counts(dropna=False, normalize=True)
train_iden.DeviceType.value_counts(dropna=False, normalize=True)
train_iden.DeviceInfo.value_counts(dropna=False, normalize=True)
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(train_iden.isnull(), cbar=False)
cor=train_iden.corr()
cor
a=cor.abs()<1
b=cor.abs()>0.5
cor.abs()[a&b]
#cor.columns.apply(lambda x:x if (x>0.9)|(x<-0.9) else nan)
# take a copy of train_iden for making comparation later
c_train_iden=train_iden
# filling Nan values with mean value

for i in [2,5,6,11,13,17,19,20]:
    
    if i<10:
        x='id_0'+str(i)
    else:
        x='id_'+str(i)
        
    c_train_iden[x].fillna(c_train_iden[x].mean(),inplace=True)
# filling Nan values based on distribution of the values in the column

for i in [3,4,9,10,14,18,32]:
    
    if i<10:
        x='id_0'+str(i)
    else:
        x='id_'+str(i)

    s = c_train_iden[x].value_counts(normalize=True)
    missing = c_train_iden[x].isnull()
    c_train_iden.loc[missing,x] = np.random.choice(s.index, size=len(c_train_iden[missing]),p=s.values)
# filling Nan values with mod

for i in [15,28,29,31,35,36,37,38]:
    
    if i<10:
        x='id_0'+str(i)
    else:
        x='id_'+str(i)
        
    c_train_iden[x].fillna(c_train_iden[x].mode()[0],inplace=True)
# filling Nan values based on distribution of the values in the column

for i in [16,30,33,34]:
    
    if i<10:
        x='id_0'+str(i)
    else:
        x='id_'+str(i)

    s = c_train_iden[x].value_counts(normalize=True)
    missing = c_train_iden[x].isnull()
    c_train_iden.loc[missing,x] = np.random.choice(s.index, size=len(c_train_iden[missing]),p=s.values)
# Filling Device Type with mod, Device info with distribution

c_train_iden['DeviceType'].fillna(c_train_iden['DeviceType'].mode()[0],inplace=True)

s = c_train_iden['DeviceInfo'].value_counts(normalize=True)
missing = c_train_iden['DeviceInfo'].isnull()
c_train_iden.loc[missing,'DeviceInfo'] = np.random.choice(s.index, size=len(c_train_iden[missing]),p=s.values)
(np.sum(pd.isnull(c_train_iden)).sort_values(ascending=False)/len(c_train_iden))*100
drop_columns=['id_07', 'id_08', 'id_21', 'id_22', 'id_24', 'id_25', 'id_26', 'id_23','id_27']
c_train_iden=c_train_iden.drop(drop_columns, axis=1)
(np.sum(pd.isnull(c_train_iden)).sort_values(ascending=False)/len(c_train_iden))*100
pd.set_option("display.max_rows", 1000, "display.max_columns", 1000)
c_train_iden.id_30.value_counts()
c_train_iden.id_30=c_train_iden.id_30.map(lambda x:'Windows' if 'Windows' in x else('iOS' if ('iOS' or 'MAC') in x else ('Linux'if 'Linux' in x else('Android' if 'Android' in x else 'Other'))))
c_train_iden.id_30.value_counts()
# d_train_iden=c_train_iden.copy()
# for i in range(len(c_train_iden.id_30)):
#     if 'Windows' in c_train_iden.iloc[i]['id_30']:
#         c_train_iden.iloc[i]['id_30']='Windows'
#     elif ('OS' in c_train_iden.iloc[i]['id_30'])|('iOS' in c_train_iden.iloc[i]['id_30']):
#         c_train_iden.iloc[i]['id_30']='OS'
#     elif 'Android' in c_train_iden.iloc[i]['id_30']:
#         c_train_iden.iloc[i]['id_30']='Android'
c_train_iden.id_31.value_counts()
c_train_iden.id_31=c_train_iden.id_31.map(lambda x:'chrome' if 'chrome' in x else('safari' if 'safari' in x else ('ie'if 'ie' in x else('edge' if 'edge' in x else('firefox' if 'firefox' in x else('samsung' if ('samsung' or 'Samsung') in x else ('opera' if 'opera' in x else 'Other')))))))
c_train_iden.id_31.value_counts()
c_train_iden.id_33.value_counts()
c_train_iden.id_33
c_train_iden.id_33=c_train_iden.id_33.map(lambda x:'Small' if int(x.split('x')[0])*int(x.split('x')[0])<480*854 else ('Medium' if int(x.split('x')[0])*int(x.split('x')[0])<1024*640 else 'Large'))
c_train_iden.id_33.value_counts()
c_train_iden.DeviceInfo.value_counts().head(100)
c_train_iden.DeviceInfo=c_train_iden.DeviceInfo.map(lambda x:'Windows' if 'Windows' in x else('Mac' if ('MacOS' or 'iOS') in x else 'Other'))
c_train_iden.DeviceInfo.value_counts()
c_train_iden.id_32.value_counts()
# print(c_train_iden.id_02.value_counts())

# print('-------------------------------')

# print(c_train_iden.id_02.min(),c_train_iden.id_02.max(),c_train_iden.id_02.nunique())

# print('-------------------------------')

# step = 100000
# max_val = c_train_iden.id_02.max()
# min_val= c_train_iden.id_02.min()
# bins = list(range(int(np.floor(min_val/step))*step,int(np.ceil(max_val/step))*step+step,step))
# clusters = pd.cut(c_train_iden.id_02,bins,labels=bins[1:])
# print(clusters.value_counts())
# c_train_iden=pd.merge(left=c_train_iden, right=target, on='TransactionID', how='left')
# c_train_iden
# sample1=c_train_iden[['id_03', 'id_04', 'id_09', 'id_10']]
# cor=sample1.corr()
# cor
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# features = ['id_03', 'id_04', 'id_09', 'id_10']
# # Separating out the features
# x = c_train_iden.loc[:, features].values
# # Separating out the target
# y = c_train_iden.loc[:,['isFraud']].values

#scale it
# x = StandardScaler().fit_transform(x)

# pca = PCA()
# principalComponents = pca.fit_transform(x)
# pca.explained_variance_ratio_
# represent=np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
# print(represent)
# plt.plot(represent)
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variances')
# plt.show()
# df_ans=pd.DataFrame({'var':pca.explained_variance_ratio_,
#                    'PC':['PC1','PC2','PC3','PC4']})
# df_ans

# sns.barplot(x='PC',y='var', data=df_ans, color='c')
# plt.ylabel('Variance Explained')
# plt.xlabel('Principle Components')
# plt.show()
# sample2=c_train_iden[['id_19', 'id_20']]
# cor=sample2.corr()
# cor
# features = ['id_19', 'id_20']
# # Separating out the features
# x = c_train_iden.loc[:, features].values
# # Separating out the target
# y = c_train_iden.loc[:,['isFraud']].values

# x = StandardScaler().fit_transform(x)

# pca = PCA()
# principalComponents = pca.fit_transform(x)

# pca.explained_variance_ratio_

# df_ans=pd.DataFrame({'var':pca.explained_variance_ratio_,
#                    'PC':['PC1','PC2']})
# sns.barplot(x='PC',y='var', data=df_ans, color='c')
# plt.ylabel('Variance Explained')
# plt.xlabel('Principle Components')
# plt.show()
# (np.sum(pd.isnull(c_train_iden)).sort_values(ascending=False)/len(c_train_iden))*100
# c_train_iden.info()
#Kategorik veriler:
#id_12,id_15,id_16,id_28,id_29,id_30,id_31,id_33,id_34,id_35,id_36,id_37,id_38,DeviceInfo,DeviceType
for i in ['id_12', 'id_15', 'id_16','id_28', 'id_29', 'id_30','id_31','id_33','id_34', 'id_35', 'id_36', 'id_37', 'id_38','DeviceInfo','DeviceType']:
    df=c_train_iden[i].value_counts(dropna=False).reset_index().rename(columns={'index':'title',i:'value'})
    sns.barplot(x='title', y='value', data=df).set_title(i)
    plt.show()
# def create_dummies( df, colname ):
#     col_dummies = pd.get_dummies(df[colname], prefix=colname)
    
#     df = pd.concat([df, col_dummies], axis=1)
#     df.drop( colname, axis = 1, inplace = True )
#     return df

# colname= ['id_12', 'id_15', 'id_16','id_28', 'id_29', 'id_30','id_31','id_33','id_34', 'id_35', 'id_36', 'id_37', 'id_38','DeviceInfo','DeviceType']
# create_dummies(c_train_iden,colname)
def frekans(data,columns,n_label="NONE"):
    
    for col in columns:
        data[col].fillna(n_label,inplace=True)
        fq_encode = data[col].value_counts(dropna=False).to_dict()   
        data[col+"_Fr"] = data[col].map(fq_encode)
        data=data.drop(col,axis=1)
    return data

columns= ['id_12', 'id_15', 'id_16','id_28', 'id_29', 'id_30','id_31','id_33','id_34', 'id_35', 'id_36', 'id_37', 'id_38','DeviceInfo','DeviceType']
c_train_iden=frekans(c_train_iden,columns)
c_train_iden.head()
#### Ersinin kisim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import datetime


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
## Function to reduce the DF size
# https://www.kaggle.com/kabure/almost-complete-feature-engineering-ieee-data
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):
    pca = PCA(n_components=n_components, random_state=rand_seed)

    principalComponents = pca.fit_transform(df[cols])

    principalDf = pd.DataFrame(principalComponents)

    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, principalDf], axis=1)
    
    print(pca.explained_variance_ratio_)
    
    return df
# CALISACAGIM DATASET
train_transaction=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID')
test_transaction=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv',index_col='TransactionID')
# test_transaction setinde isFraud sutunu bulunmadigindan concat oncesi yeni ekliyoruz ve 2 degerini dolduruyoruz
test_transaction['isFraud']=2
train_transaction = reduce_mem_usage(train_transaction)
test_transaction = reduce_mem_usage(test_transaction)
# train_transaction ve test_transaction setlerini concat ediyoruz
transaction=pd.concat([train_transaction, test_transaction], axis=0, sort=False )
transaction = transaction.reset_index()
transaction = reduce_mem_usage(transaction)
# BENIM KISIM OLAN COLUMNLARI BASKA BIR VARIABLE ATIYORUZ ve CALISMAMIZDA COPYASINI KULLANIYORUZ
transaction_mycolumns=transaction.iloc[:,0:55]
my_transaction=transaction_mycolumns.copy()
print('Shape before PCA')
my_transaction.shape
# Eksik verileri oransal bar ile gorsellestirme
fig, ax = plt.subplots(figsize=(30,20)) 
sns.heatmap(my_transaction.isnull(), cbar=False)
# Kolonlarda ne kadar eksik veri var - bunun yuzdesini  nedir --> yuksekten asagi siraladik(TAM OLANLAR HARIC)

mis_value = my_transaction.isnull().sum()
mis_value_percent = 100*my_transaction.isnull().sum()/len(my_transaction)
mis_dtype = my_transaction.dtypes

mis_value_table = pd.concat([mis_value,mis_value_percent,mis_dtype], axis = 1)
mis_value_table.columns=['count', 'percent','type']
mis_value_table = mis_value_table.sort_values('percent',ascending=False)
mis_value_table = mis_value_table[mis_value_table['percent']>0]
pd.set_option('display.max_rows', None)
mis_value_table
# % 85 den fazla Eksik Veri olan 5 sutunu siliyoruz
# dist2,D7 D8 D9 D12

mis_value_table_per80 = mis_value_table[mis_value_table['percent']>85]
drop_index_column_name=mis_value_table_per80.index[:]

for column_name in drop_index_column_name:
    my_transaction.drop(column_name, axis=1, inplace=True)
my_transaction.columns
c_feat = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
              'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

for col in c_feat:
    my_transaction[col] = my_transaction[col].fillna((my_transaction[col].min() - 1))
    my_transaction[col] = (minmax_scale(my_transaction[col], feature_range=(0,1)))
  
my_transaction = PCA_change(my_transaction, c_feat, prefix='PCA_C_', n_components=3)

c_features = ['PCA_C_0', 'PCA_C_1', 'PCA_C_2']

km = KMeans(n_clusters=4)
km = km.fit(my_transaction[c_features])
my_transaction['clusters_C'] = km.predict(my_transaction[c_features])
my_transaction = reduce_mem_usage(my_transaction)
my_transaction.shape
d_features = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6','D10', 'D11', 'D13', 'D14', 'D15']

one_fill = ['D1', 'D2', 'D3', 'D5', 'D6','D10', 'D11', 'D13', 'D14']

nn_fill = ['D4', 'D15']

for col in one_fill:
    my_transaction[col] = (minmax_scale(my_transaction[col], feature_range=(0,1)))
    my_transaction[col] = my_transaction[col].fillna(-1)
    
for col in nn_fill:
    my_transaction[col] = (minmax_scale(my_transaction[col], feature_range=(0,1)))
    my_transaction[col] = my_transaction[col].fillna(-1)
  


my_transaction = PCA_change(my_transaction, d_features, prefix='PCA_D_', n_components=8)

pca_d = ['PCA_D_0', 'PCA_D_1', 'PCA_D_2', 'PCA_D_3',
         'PCA_D_4', 'PCA_D_5', 'PCA_D_6', 'PCA_D_7']

km = KMeans(n_clusters=8)
km = km.fit(my_transaction[pca_d])
my_transaction['clusters_D'] = km.predict(my_transaction[pca_d])
my_transaction = reduce_mem_usage(my_transaction)
my_transaction.shape
my_transaction.head()
my_transaction['addr1'] = my_transaction['addr1'].fillna(0)
my_transaction['addr2'] = my_transaction['addr2'].fillna(0)

my_transaction['diff_adrr'] = my_transaction.addr1 - my_transaction.addr2
my_transaction['diff_adrr_plus'] = my_transaction.addr1 + my_transaction.addr2

my_transaction['first_value_addr1'] = my_transaction['addr1'].astype(str).str[0:1].astype(float)
my_transaction['two_value_addr1'] = my_transaction['addr1'].astype(str).str[0:2].astype(float)
## Filling Dist1 Nan's
my_transaction['dist1'] = my_transaction['dist1'].fillna(-1)
# card1 de missing value yok 
# card2 yi inceleyelim 1.603 nan degeri median(Ortanca Değer) ile dolduruyoruz
my_transaction.card2.describe()
my_transaction['card2'].fillna(my_transaction['card2'].median(),inplace=True)
# card3 4567 Nan degerini modu olan 150 degeri ile dolduruyoruz %87 si bu degerde
my_transaction.card3.fillna(my_transaction['card3'].mode()[0],inplace=True)
# card4 ve card6 Kategorik degerleri incelerken dolduracagiz
# card5 8806 nan degeri median() ile dolduracagiz
my_transaction['card5'].fillna(my_transaction['card5'].median(),inplace=True) 
my_transaction = reduce_mem_usage(my_transaction)
my_transaction.shape
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    my_transaction[c + '_bin'] = my_transaction[c].map(emails)
    my_transaction[c + '_suffix'] = my_transaction[c].map(lambda x: str(x).split('.')[-1])
    my_transaction[c + '_suffix'] = my_transaction[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    my_transaction.drop(c, axis=1, inplace=True)
# Kategorik degerlerin eksik degerlerini NONE ile doldurma 
# ve sonrasinda kategorik degiskeni Frekansi ile numerik hale getirme
def frekans(data,columns,n_label="NONE"):
    
    for col in columns:
        data[col].fillna(n_label,inplace=True)
        fq_encode = data[col].value_counts(dropna=False).to_dict()   
        data[col+"_Fr"] = data[col].map(fq_encode)
        data=data.drop(col,axis=1)
    return data
M_columns = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']

my_transaction=frekans(my_transaction,M_columns)
my_transaction.card4.replace('american express','other',inplace=True)
my_transaction.card4.replace('discover','other',inplace=True)
my_transaction=frekans(my_transaction,["card4"],n_label="other")
my_transaction.card6.replace('debit or credit','debit',inplace=True)
my_transaction.card6.replace('charge card','debit',inplace=True)
my_transaction=frekans(my_transaction,["card6"],n_label="debit")
object_columns_name=my_transaction.select_dtypes(include='object').columns
my_transaction=frekans(my_transaction,['ProductCD'])
mail_columns = ['P_emaildomain_bin','P_emaildomain_suffix','R_emaildomain_bin','R_emaildomain_suffix']
my_transaction=frekans(my_transaction,mail_columns)
for i in my_transaction.columns:
    my_transaction[i].value_counts(normalize=True)
    print(i)
    print(my_transaction[i].value_counts(normalize=True))
    print ('\n')
my_transaction = reduce_mem_usage(my_transaction)
my_transaction.shape
my_transaction.info()
# Preprocess date column
START_DATE = '2017-12-01'
my_transaction = my_transaction.rename(columns={'TransactionDT': 'TransactionDate'})
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
my_transaction['TransactionDate'] = my_transaction['TransactionDate'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


print(my_transaction['TransactionDate'].head())
print(my_transaction['TransactionDate'].tail())
import seaborn as sns
my_transaction_TransactionAmt_Fraud=my_transaction[['isFraud','TransactionAmt']]
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(my_transaction_TransactionAmt_Fraud.corr(), ax=ax,linewidths=.5,annot=True)
plt.show()
# boxplot
sns.boxplot(x=my_transaction['TransactionAmt']);
# boxplot da esik deger atama
Q1=my_transaction['TransactionAmt'].quantile(0.25)
Q3=my_transaction['TransactionAmt'].quantile(0.75)
IQR=Q3-Q1
my_TransactionAmt_Alt_Sinir = Q1-2.5*IQR
my_TransactionAmt_Ust_Sinir = Q3 + 2.5*IQR

Q1, Q3 , IQR,my_TransactionAmt_Alt_Sinir,my_TransactionAmt_Ust_Sinir
# boxplot ile belirlenen aykiri degerlere erismek
aykiri_TF=(my_transaction['TransactionAmt']<my_TransactionAmt_Alt_Sinir)|(my_transaction['TransactionAmt']>my_TransactionAmt_Ust_Sinir)
my_transaction[aykiri_TF].index
# baskilama ile ust deger sonrasi degerlere ust degeri, alt deger sonrasina ise alt degeri atama
def AykiriDegeriBaskila(deger):
    
    if deger > my_TransactionAmt_Ust_Sinir:
       
        deger=my_TransactionAmt_Ust_Sinir
    elif deger < my_TransactionAmt_Alt_Sinir:
       
        deger=-my_TransactionAmt_Alt_Sinir
    
    return deger

my_transaction['TransactionAmt'] = my_transaction['TransactionAmt'].apply(lambda x: AykiriDegeriBaskila(x))
# Hakanin kisim
import numpy as np
import pandas as pd


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import seaborn as sns

import gc
def PCA_(df, cols, prefix='PCA_', rand_seed=4):
    pca = PCA(random_state=rand_seed)
    pca.fit_transform(df[cols])
    represent=np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
    print(represent)
    n_components=0
    for i in represent:
        
        n_components+=1
        if i >=98:
            print("n_components= ",n_components)
            break
            
    pca = PCA(random_state=rand_seed,n_components=n_components)
    principalComponents = pca.fit_transform(df[cols])
    
    principalDf = pd.DataFrame(principalComponents)

    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, principalDf], axis=1)
    print(pca.explained_variance_ratio_)
    return df
transaction_V = transaction.iloc[:,55:]
V_columns= transaction_V.columns

for col in V_columns:
    transaction_V[col] = transaction_V[col].fillna((transaction_V[col].min() - 1))
    transaction_V[col] = (minmax_scale(transaction_V[col], feature_range=(0,1)))
transaction_V=PCA_(transaction_V,V_columns,prefix='PCA_V_')
my_transaction.drop(['clusters_C','clusters_D'], axis=1, inplace=True)
my_transaction.head()
print(len(my_transaction))
transaction_V.head()
print(len(transaction_V))
transaction_V.info()
sum_trans=pd.concat([my_transaction, transaction_V], axis=1)
del my_transaction
del transaction_V
sum_trans = reduce_mem_usage(sum_trans)
sum_trans.head(100)
c_train_iden.head(100)
final_data=pd.merge(left=sum_trans, right=c_train_iden, on='TransactionID', how='left')
final_data.head(100)
final_data.tail(100)