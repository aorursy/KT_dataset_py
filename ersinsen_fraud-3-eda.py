# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# CALISACAGIM DATASET
train_transaction=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_transaction=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
# test_transaction setinde isFraud sutunu bulunmadigindan concat oncesi yeni ekliyoruz ve 2 degerini dolduruyoruz
test_transaction['isFraud']=2
# train_transaction ve test_transaction setlerini concat ediyoruz
transaction=pd.concat([train_transaction, test_transaction], ignore_index=True)
transaction['D1'].head(40)
# Column ve Degerlerine genel bakis
for col, values in transaction.iteritems():
    num_uniques = values.nunique()
    print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))
    print (values.unique())
    print ('\n')
# NE KADAR ROW-COLINM OLDUGU

print(f'Train Transaction has {transaction.shape[0]} rows and {transaction.shape[1]} columns.')
#COLUMN ISIMLER
transaction.columns
# BENIM KISIM OLAN COLUMNLARI BASKA BIR VARIABLE ATIYORUZ ve CALISMAMIZDA COPYASINI KULLANIYORUZ
transaction_mycolumns=transaction.iloc[:,0:55]
my_transaction=transaction_mycolumns.copy()
my_transaction.info()
my_transaction.info(memory_usage='deep')
my_transaction.describe()
# kololasyonu anlam ifade eden kisimlari gozlemleyelim
a=my_transaction.corr().abs()<1
b=my_transaction.corr().abs()>0.5
my_transaction.corr().abs()[a&b]

# NE KADAR COLUMNDA MISSING VALUE OLDUGU
print(f'There are {my_transaction.isnull().any().sum()} columns in my_train transaction dataset with missing values.')
# Eksik verileri oransal bar ile gorsellestirme
msno.bar(train_Mytransaction);
msno.matrix(my_transaction);
# degiskenler arasindaki yokluk koralasyonu  -1 demek- birinde null varsa digerinde de null her durumda var
msno.heatmap(my_transaction);
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

for column_name in ['P_emaildomain', 'R_emaildomain']:
    my_transaction.drop(column_name, axis=1, inplace=True)
object_columns_name=my_transaction.select_dtypes(include='object').columns
object_columns_name
for kategorik_sutun in object_columns_name:
    modun_stringdegeri=my_transaction[kategorik_sutun].mode()[0]
    my_transaction[kategorik_sutun].fillna(modun_stringdegeri,inplace=True)
# Kategorik degerlerin eksik degerlerini doldurma yontemlerinden Frekansi kullanacagiz
def frekans(data,columns,n_label="NONE"):
    
    for col in columns:
        #data[col].fillna(n_label,inplace=True)
        fq_encode = data[col].value_counts(dropna=False).to_dict()   
        data[col+"_Fr"] = data[col].map(fq_encode)
        data=data.drop(col,axis=1)
    return data
my_transaction = frekans(my_transaction,object_columns_name)
my_transaction.head()
my_transaction.info()
# Preprocess date column
START_DATE = '2017-12-01'
my_transaction = my_transaction.rename(columns={'TransactionDT': 'TransactionDate'})
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
my_transaction['TransactionDate'] = my_transaction['TransactionDate'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


print(my_transaction['TransactionDate'].head())
print(my_transaction['TransactionDate'].tail())
my_transaction.columns
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
my_transaction.card1.describe()
df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 2].drop('isFraud', axis=1)