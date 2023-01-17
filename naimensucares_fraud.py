# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
import os
import gc


test_ID=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
train_ID=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_TR=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
# train_TR=pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
train_df=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
train_df.info()
train_ID.head()

train_ID.info()
## Function to reduce the DF size
def memory(df, verbose=True):
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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


data=memory(train_df)

del train_df
data.head()

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
    
    return df
def frekans(data,columns,n_label="NONE"):
    
    for col in columns:
        data[col].fillna(n_label,inplace=True)
        fq_encode = data[col].value_counts(dropna=False).to_dict()   
        data[col+"_Fr"] = data[col].map(fq_encode)
        data=data.drop(col,axis=1)
    return data

data.head()
V_columns=data.columns[55:]
s=data.loc[:,V_columns].head()
!pip install missingno
import missingno as msno
msno.bar(data.loc[:,V_columns]);

for col in V_columns:
    data[col] = data[col].fillna((data[col].min() - 1))
    data[col] = (minmax_scale(data[col], feature_range=(0,1)))
data=PCA_(data,V_columns,prefix='PCA_V_')
    
M_columns = ['M1','M2','M3','M5','M6','M7','M8','M9']
data.loc[:,M_columns].head()
for col in M_columns:
    print(data[col].value_counts())
    print("NaN",data[col].isnull().sum())
    print("****************")

M_columns = ['M1','M2','M3','M5','M6','M7','M8','M9']

data=frekans(data,M_columns)


data.head()
M_fr=list(data.columns[-len(M_columns):])
for col in M_fr:
    data[col] = (minmax_scale(data[col], feature_range=(0,1)))
data=PCA_(data,M_fr,prefix="PCA_M_")


data.head()
# i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

# for df in [df]:
#     df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
#     df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
C_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

data.loc[:,C_columns].head()
msno.bar(data.loc[:,C_columns]);
C_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']

for col in C_columns:
    data[col] = data[col].fillna((data[col].min() - 1))
    data[col] = (minmax_scale(data[col], feature_range=(0,1)))

data=PCA_(data,C_columns,prefix='PCA_C_')
data.head()
D_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7','D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14','D15']

data.loc[:,D_columns].head()

msno.bar(data.loc[:,D_columns]);
D_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7','D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14','D15']


for col in D_columns:
    data[col] = (minmax_scale(data[col], feature_range=(0,1)))
    data[col] = data[col].fillna(-1)
data=PCA_(data,D_columns,prefix='PCA_D_')
data.head()
data.M4.value_counts()
for col in ["M4"]:
    print(data[col].value_counts())
    print("NaN",data[col].isnull().sum())
    print("****************")
col=["M4"]
data=frekans(data,col)
data.head()
data=memory(data)
df=memory(train_ID)
new=data.merge(df,how='left', 
               left_index=True, 
               right_index=True, on='TransactionID')
del df
del data
new.head()
ID_columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05','id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11']
new.loc[:,ID_columns].head()

ID_columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05','id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11']
    
for col in ID_columns:
    new[col] = (minmax_scale(new[col], feature_range=(0,1)))
    new[col].fillna(-1, inplace=True)
    
new=PCA_(new,ID_columns,prefix='PCA_id_')
new.head()
id_mix_columns=['id_12','id_15','id_16','id_27','id_28','id_29','id_23']
new.loc[:,id_mix_columns].head()

for col in id_mix_columns:
    
    print(new[col].value_counts())
    print("NaN ",new[col].isnull().sum())
    print("****************")
    
id_mix_columns=['id_12','id_15','id_16','id_27','id_28','id_29','id_23']
new=frekans(new,id_mix_columns)
new.head()
id_mix_2=['id_35','id_36','id_37','id_38']
new.loc[:,id_mix_2].head()
for col in id_mix_2:
    print(new[col].value_counts())
    print("NaN",new[col].isnull().sum())
    print("****************")
    
id_mix_2=['id_35','id_36','id_37','id_38']
new=frekans(new,id_mix_2)

new.head()
id_mix_3=['id_13','id_14', 'id_17', 'id_18', 'id_19', 'id_20','id_21', 'id_22','id_24', 'id_25', 'id_26']
new.loc[:,id_mix_3].head()

msno.bar(new.loc[:,id_mix_3]);
id_mix_3=['id_13','id_14', 'id_17', 'id_18', 'id_19', 'id_20','id_21', 'id_22','id_24', 'id_25', 'id_26']
for col in id_mix_3: 
    new[col].fillna(new[col].min()-100, inplace=True)
    new[col] = (minmax_scale(new[col], feature_range=(0,1)))
    

new=PCA_(new,id_mix_3,prefix='PCA_id_13_26_')
new.head()
for col in ["DeviceInfo"]:
    print(new[col].value_counts())
    print("NaN",new[col].isnull().sum())
    print("****************")
    
new.rename(columns={'DeviceInfo':"device_name"},inplace=True)

new.loc[new['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung' 
new.loc[new['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung' 
new.loc[new['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung' 
new.loc[new['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola' 
new.loc[new['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola' 
new.loc[new['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola' 
new.loc[new['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG' 
new.loc[new['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV' 
new.loc[new['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei' 
new.loc[new['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei' 
new.loc[new['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei' 
new.loc[new['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE' 
new.loc[new['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE' 
new.loc[new['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux' 
new.loc[new['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony' 
new.loc[new['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC' 
new.loc[new['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

new.loc[new.device_name.isin(new.device_name.value_counts()[new.device_name.value_counts() < 200].index), 'device_name'] = "Others"
new.device_name.value_counts()
col=["device_name"]
new=frekans(new,col)
new.head()
new.loc[new['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows' 
new.loc[new['id_30'].str.contains('Mac', na=False),  'id_30'] = 'Mac' 
new.loc[new['id_30'].str.contains('iOS', na=False),  'id_30'] = 'iOS' 
new.loc[new['id_30'].str.contains('Android', na=False),  'id_30'] = 'Android'

new.id_30.unique()
new.id_30.value_counts()
col=["id_30"]
new=frekans(new,col)
new.head()
new.id_31.unique()

new.loc[new['id_31'].str.contains('amsung', na=False), 'id_31'] = 'Samsung' 
new.loc[new['id_31'].str.contains('chrom', na=False), 'id_31'] = 'Chrome'  
new.loc[new['id_31'].str.contains('ndroid', na=False), 'id_31'] = 'Chrome' 
new.loc[new['id_31'].str.contains('google', na=False), 'id_31'] = 'Chrome' 
new.loc[new['id_31'].str.contains('icrosoft', na=False), 'id_31'] = 'Microsoft'
new.loc[new['id_31'].str.contains('edge', na=False), 'id_31'] = 'Microsoft'
new.loc[new['id_31'].str.contains('ie', na=False), 'id_31'] = 'Microsoft'
new.loc[new['id_31'].str.contains('opera', na=False), 'id_31'] = 'Opera'
new.loc[new['id_31'].str.contains('safari', na=False), 'id_31'] = 'Safari'  
new.loc[new['id_31'].str.contains('fox', na=False), 'id_31'] = 'Firefox'

new.loc[new.id_31.isin(new.id_31.value_counts()[new.id_31.value_counts() < 400].index), 'id_31'] = "Others"

new.id_31.unique()
new.id_31.value_counts()
col=["id_31"]
new=frekans(new,col)
new.head()
new.DeviceType.unique()
new.DeviceType.value_counts()
col=["DeviceType"]
new=frekans(new,col)
new.head()
new.id_32.value_counts()
col=["id_32"]
new=frekans(new,col)
new.head()
new.id_34.unique()
col=["id_34"]
new=frekans(new,col)
new.head()
new['screen_width'] = new['id_33'].str.split('x', expand=True)[0] 
new['screen_height'] = new['id_33'].str.split('x', expand=True)[1] 
new['screen_width'].fillna(-1, inplace=True) 
new['screen_height'].fillna(-1, inplace=True) 
new.drop('id_33', axis=1, inplace=True)
new.head()
df=memory(new)
del new
df.head()
# data.select_dtypes(include=['object'])

data=df.loc[:,'TransactionID':'R_emaildomain']
msno.bar(data);

msno.matrix(data);
msno.heatmap(data)
percent_missing = data.isnull().sum() * 100 / len(data)
percent_missing
del df['dist2']
df.card1.describe()
df.card2.describe()
df['card2'].value_counts()    #belli bir sayida yigilma yok
import matplotlib.pyplot as plt

plt.scatter(data.TransactionID, df.card2,s=0.00001)
plt.show()
df['card2'].fillna(df['card2'].median(),inplace=True)
df.card2.describe()
df['card3'].value_counts()
df['card3'].mode()
(df['card3']==df['card3'].mode()[0]).sum()*100/(len(data))
df.card3.fillna(df['card3'].mode()[0],inplace=True)
df['card4'].value_counts()
import numpy as np
import matplotlib.pyplot as plt
fraud0_visa=len(data[(data.card4=='visa')&(data.isFraud==0)])
fraud0_master=len(data[(data.card4=='master')&(data.isFraud==0)])
fraud0_american=len(data[(data.card4=='american express')&(data.isFraud==0)])
fraud0_disc=len(data[(data.card4=='discover')&(data.isFraud==0)])

fraud1_visa=len(data[(data.card4=='visa')&(data.isFraud==1)])
fraud1_master=len(data[(data.card4=='master')&(data.isFraud==1)])
fraud1_american=len(data[(data.card4=='american express')&(data.isFraud==1)])
fraud1_disc=len(data[(data.card4=='discover')&(data.isFraud==1)])
# data to plot
n_groups = 4
isFraud_0= (fraud0_visa, fraud0_master, fraud0_american, fraud0_disc)
isFraud_1 = (fraud1_visa, fraud1_master, fraud1_american, fraud1_disc)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, isFraud_0, bar_width,
alpha=opacity,
color='b',
label='isFraud=0')

rects2 = plt.bar(index + bar_width, isFraud_1, bar_width,
alpha=opacity,
color='g',
label='isFraud=1')

plt.xlabel('Card Type')
plt.ylabel('Scores')
plt.title('Scores by card type')
plt.xticks(index + bar_width, ('visa', 'mastercard', 'americanexpress', 'discover'))
plt.legend()

plt.tight_layout()
plt.show()

print('bos ve dolandiricilik:',len(data[(data.card4=='naN')&(data.isFraud==1)]))
print('bos ve dolandiricilik yok;',len(data[(data.card4=='naN')&(data.isFraud==0)]))
print('tum dolandiricilik;',len(data[(data.isFraud==1)]))
print('-------------------------')
print('visa ve dolandiricilik',len(data[(data.card4=='visa')&(data.isFraud==1)]))
print('master ve dolandiricilik',len(data[(data.card4=='mastercard')&(data.isFraud==1)]))
print('american ve dolandiricilik',len(data[(data.card4=='american express')&(data.isFraud==1)]))
print('discover ve dolandiricilik',len(data[(data.card4=='discover')&(data.isFraud==1)]))
df.card4.unique()
df.card4.value_counts()
col=['card4']
df=frekans(df,col)
df.head()
df['card5'].value_counts()

df['card5'].mode()
(df['card5']==df['card5'].mode()[0]).sum()*100/(len(df))
df.card5.describe()
df['card5'].fillna(df['card5'].median(),inplace=True)  #%0.72 si bostu
df['card5'].isnull().sum()
df.card5.describe()
df['card6'].value_counts()
data.card6.replace('debit or credit','debit',inplace=True)
data.card6.replace('charge card','debit',inplace=True)
(df['card6'].isnull().sum())*100/(len(df))
col=["card6"]
df=frekans(df,col,n_label="debit")
df.head()
#simdilik mail adreslerinin ayni veya farkli oldugunu belirten ek column
df['the_same']=np.where(df['P_emaildomain'] == df['R_emaildomain'],True,False)  
df.the_same.value_counts()
diff_addr_Fraud=len(df[(df.the_same== False) & (df.isFraud==1)])
diff_addr_Fraud
diff_addr_Fraud*100/len(df.loc[(df.isFraud==1)])
x=len(df[(df.the_same== False) & (df.isFraud==1)]) /len(df)
x
y=len(df[(df.isFraud==1)])/len(df)
y
x/y 
x=len(df[(df.the_same== False) & (df.isFraud==0)]) /len(df)
x
y=len(df.loc[(df.isFraud==0)])/len(df)
y
x/y  
del df['the_same']
df["P_emaildomain"].value_counts()
df.loc[df['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

df.loc[df['P_emaildomain'].isin(["yahoo.co.jp", "yahoo.co.uk","yahoo.com",
                                     "yahoo.com.mx", "yahoo.de", "yahoo.es", "yahoo.fr",
                                     "ymail.com","frontier.com", "frontiernet.net", 
                                     "rocketmail.com"]), 'P_emaildomain'              ] = 'Yahoo Mail'

df.loc[df['P_emaildomain'].isin(["hotmail.co.uk", "hotmail.com", "hotmail.de", 
                                     "hotmail.es","hotmail.fr", "live.com", "live.com.mx", 
                                     "live.fr","msn.com","outlook.com", "outlook.es"  ]), 
                                     'P_emaildomain'                                  ] = 'Microsoft'

df.loc[df['P_emaildomain'].isin(["icloud.com","mac.com","me.com"]),'P_emaildomain'] = "Apple"

df.loc[df['P_emaildomain'].isin(["att.net", "prodigy.net.mx", "sbcglobal.net"
]), 'P_emaildomain'                                                                   ] = "AT&T"

df.loc[df['P_emaildomain'].isin(["centurylink.net", "embarqmail.com","q.com"]), 
                                     'P_emaildomain'                                  ] = "Centurylink"

df.loc[df['P_emaildomain'].isin(["aim.com", "aol.com"]), 'P_emaildomain'          ] = "AOL"

df.loc[df['P_emaildomain'].isin(["charter.net","twc.com"]), 'P_emaildomain'       ] = "Spectrum"

df.loc[df.P_emaildomain.isin(df.P_emaildomain.value_counts()
                                 [df.P_emaildomain.value_counts() <= 500 ]\
                                         .index), 'P_emaildomain'                     ] = "Others"
df.P_emaildomain.fillna("NoInf", inplace=True)
df["P_emaildomain"].unique()
df["P_emaildomain"].value_counts()
col=["P_emaildomain"]
df=frekans(df,col)
df.head()
df.R_emaildomain.value_counts()

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
          'scranton.edu': 'other', 'netzero.net': 'other',
          'optonline.net': 'other', 'comcast.net': 'other', 
          'cfl.rr.com': 'other', 'sc.rr.com': 'other',
          'suddenlink.net': 'other', 'windstream.net': 'other',
          'gmx.de': 'other', 'earthlink.net': 'other', 
          'servicios-ta.com': 'other', 'bellsouth.net': 'other', 
          'web.de': 'other', 'mail.com': 'other',
          'cableone.net': 'other', 'roadrunner.com': 'other', 
          'protonmail.com': 'other', 'anonymous.com': 'other',
          'juno.com': 'other', 'ptd.net': 'other',
          'netzero.com': 'other', 'cox.net': 'other', 
          'hotmail.co.uk': 'microsoft', 
          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 
          'live.com': 'microsoft', 'aim.com': 'aol',
          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 
          'hotmail.com': 'microsoft',  
          'hotmail.fr': 'microsoft',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 
          'yahoo.de': 'yahoo', 
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 
          'msn.com': 'microsoft', 'q.com': 'centurylink',
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 
           'rocketmail.com': 'yahoo', 
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 
          'embarqmail.com': 'centurylink', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo',
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
           'aol.com': 'aol', 'icloud.com': 'apple'}
df["new_R_emaildomain"] = df["R_emaildomain"].map(emails)
df.new_R_emaildomain.unique()
df.new_R_emaildomain.value_counts()
del df["R_emaildomain"]
col=["new_R_emaildomain"]
df=frekans(df,col)
df.head()
df.dist1.value_counts()
df.dist1.describe()
sns.boxplot(x=df.dist1)
import seaborn as sns
data_dist1_Fraud=df[['isFraud','dist1']]
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(data_dist1_Fraud.corr(), ax=ax,linewidths=.5,annot=True)
plt.show()
Q1=df.dist1.quantile(0.25)
Q3=df.dist1.quantile(0.75)
IQR=Q3-Q1
altsinir=Q1-1.5*IQR
ustsinir=Q3+1.5*IQR
print('altsinir:',altsinir)
print('ustsinir:',ustsinir)
def Quantile(x):
    
    if x > ustsinir:
       
        x=ustsinir
    elif x < altsinir:
       
        x=-altsinir
    
    return x

u=df["dist1"].apply(lambda x: Quantile(x))
df["dist1"]=u
df.head()
sns.boxplot(x=df.dist1)
df.dist1.describe()
df.dist1.fillna(df.dist1.mean(),inplace=True)

df.dist1.isnull().sum()
df.head()
df.dist1.describe()
sns.boxplot(x=df.dist1)
df.head()
import seaborn as sns
data_dist1_Fraud=df[['isFraud','dist1']]
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(data_dist1_Fraud.corr(), ax=ax,linewidths=.5,annot=True)
plt.show()

df.TransactionDT.value_counts()
import datetime
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
df['_Weekdays'] = df['Date'].dt.dayofweek
df['_Hours'] = df['Date'].dt.hour
df['_Days'] = df['Date'].dt.day
df.head()
df.ProductCD.value_counts()
col=["ProductCD"]
df=frekans(df,col)
df.head()
df.screen_height.value_counts()
df.loc[df.screen_height.isin(df.screen_height.value_counts()[df.screen_height.value_counts() < 300].index), 'screen_height'] = "Others"
df.screen_height.value_counts()
col=["screen_height"]
df=frekans(df,col)
df.head()
df.screen_width.value_counts()
df.loc[df.screen_width.isin(df.screen_width.value_counts()[df.screen_width.value_counts() < 300].index), 'screen_width'] = "Others"
df.screen_width.value_counts()
col=["screen_width"]
df=frekans(df,col)
df.head()
finally_data=memory(df)
del df
X_train = finally_data.sort_values('TransactionDT').drop(['isFraud','TransactionDT', 'Date'], axis=1)
y_train = finally_data.sort_values('TransactionDT')['isFraud']
X_train.head()
cols=X_train.columns
print(cols)
X_train.shape
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score


# **************************Gridvalidation******************************
# xgb_params = {
#         'n_estimators': [100, 500, 1000, 2000],
#         'subsample': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5,6],
#         'learning_rate': [0.1,0.01,0.02,0.05],
#         "min_samples_split": [2,5,10]}


# model = XGBClassifier()
# model.fit(X_train, y_train)
# print(model)
# y_pred = model.predict(X_train)
# # predictions = [round(value) for value in y_pred]
# model_2=XGBClassifier()
# xgb_cv_model = GridSearchCV(model_2, xgb_params, cv = 10,  verbose = 2)

# xgb_cv_model.fit(X_train, y_train)

# print(xgb_cv_model.best_params_)


# Grid degerler
# Fitting 10 folds for each of 576 candidates, totalling 5760 fits
# [CV] learning_rate=0.1, max_depth=3, min_samples_split=2, n_estimators=100, subsample=0.6 
# ***************************************************************************


model = XGBClassifier(learning_rate=0.1, max_depth=3, min_samples_split=2, n_estimators=100, subsample=0.6)
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_train)


# accuracy = accuracy_score(y_train, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


scores = cross_val_score(model, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())


feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,cols)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:20])
plt.title('XGBClassifier Feature importances')
plt.tight_layout()
plt.show()

x=gc.collect()