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
df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

df
df.info()
df.columns
cat_df = df.select_dtypes(include=['int64']).copy()

cat_df = cat_df.drop(columns="ID")#delete ID from categorical data -> not useful

cat_df.columns
cat_df.shape
cat_df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}, inplace=True)
encode_columns=['SEX','MARRIAGE','EDUCATION']

for i in encode_columns:

    cat_df=pd.get_dummies(cat_df, columns=[i])
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

cat_df.columns

unique_status = np.unique(cat_df[['PAY_0']])

print("total unique statuses:", len(unique_status))

print(unique_status)
monthes=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for i in monthes:

    cat_df=pd.get_dummies(cat_df, columns=[i])

bins = [21, 30, 40, 50, 60, 76]

group_names = ['21-30', '31-40', '41-50', '51-60', '61-76']

age_cats = pd.cut(cat_df['AGE'], bins, labels=group_names)

cat_df['age_cats'] = pd.cut(cat_df['AGE'], bins, labels=group_names)
cat_df=pd.get_dummies(cat_df, columns=['age_cats'])
cat_df.columns
len(cat_df.columns)
cat_df.dtypes
len(cat_df.columns)
num_df = df.select_dtypes(include=['float64']).copy()

num_df.columns
bills=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5','BILL_AMT6']

col_names=['Q_BILL_AMT1', 'Q_BILL_AMT2', 'Q_BILL_AMT3', 'Q_BILL_AMT4','Q_BILL_AMT5', 'Q_BILL_AMT6']

i=0#counter 



for col in bills:

    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)

    i+=1

    

num_df.columns
num_df.head()
pays=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6','LIMIT_BAL']

col_names=['Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']

i=0#counter 



for col in pays:

    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]

    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)

    i+=1

    

num_df.columns
encode_columns=['Q_BILL_AMT1', 'Q_BILL_AMT2','Q_BILL_AMT3', 'Q_BILL_AMT4', 'Q_BILL_AMT5', 'Q_BILL_AMT6','Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']

for i in encode_columns:

    num_df=pd.get_dummies(num_df, columns=[i])
num_df.head()
num_df.columns
len(num_df.columns)
num_df['late_payer']=df['PAY_0'].apply(lambda x: 1 if x > 1 else 0)



num_df['late_payer'].head()
bill_mons=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

cols=['OVER_BILL_AMT1','OVER_BILL_AMT2','OVER_BILL_AMT3','OVER_BILL_AMT4','OVER_BILL_AMT5','OVER_BILL_AMT6']

i=0#counter



for mon in bill_mons:

    num_df[cols[i]]=df[mon].apply(lambda x: 1 if x < 0 else 0)

    i+=1

    

num_df['OVER_BILL_AMT1'].head()    
data = pd.concat([cat_df, num_df], axis=1)
target=data['default.payment.next.month']

data = data.drop(columns='default.payment.next.month')#delete target from dataframe
data.head()
len(data.columns)
#data.to_csv('mycsvfile.csv',index=False)