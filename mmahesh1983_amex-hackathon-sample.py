# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import plot_tree, DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

rndval = 42
df_camp=pd.read_csv('/kaggle/input/amex-analytics-vidya/campaign_data.csv')

df_test=pd.read_csv('/kaggle/input/amex-analytics-vidya/test_QyjYwdj.csv')

df_coup = pd.read_csv('/kaggle/input/amex-analytics-vidya/coupon_item_mapping.csv')

df_item=pd.read_csv('/kaggle/input/amex-analytics-vidya/item_data.csv')

df_train=pd.read_csv('/kaggle/input/amex-analytics-vidya/train.csv')

df_ctran=pd.read_csv('/kaggle/input/amex-analytics-vidya/customer_transaction_data.csv')

df_demog=pd.read_csv('/kaggle/input/amex-analytics-vidya/customer_demographics.csv')
#Creating columns to join train and test

df_test['redemption_status'] = 0

df_test['data_type'] = 'test'

df_train['data_type'] = 'train'
def first_stage(df):

    df = df.copy()

    df['family_size'] = df['family_size'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)

    return df



def simple_impute(df, imputer, cols, test=False):

    df = df.copy()

    if not test:

        imputer.fit(df[cols])

    df[cols] = imputer.transform(df[cols])

    return df



def impute_DemogData(df,test=False):

    df = df.copy()

    if not test:

        df.loc[(df.family_size == '1') & (df.marital_status.isnull()),'marital_status'] = 'Single'

        df.loc[(df.marital_status.isnull()) & (~df.no_of_children.isnull()),'marital_status'] = 'Married'

        df.loc[(df.marital_status.isnull()) & (df.age_range == '26-35'),'marital_status'] = 'Single'

        df.loc[(df.marital_status.isnull()) & (df.age_range != '26-35'),'marital_status'] = 'Married'

        df.loc[(df.marital_status == 'Married') & (df.family_size == '2') & (df.no_of_children.isnull()),'no_of_children'] = '0'

        df.loc[(df.marital_status == 'Single') & (df.no_of_children.isnull()),'no_of_children'] = '-1'

        return df

    

orde = OrdinalEncoder()



def EncodeColumns(df, encoder, cols, test=False):

    df = df.copy()

    if not test:

        encoder.fit(df[cols])

    df[cols] = encoder.transform(df[cols])

    return df



df_demog = impute_DemogData(df_demog,False)

df_check = pd.concat([df_test,df_train],axis=0)
df_check.redemption_status.sum()
df_check1 = df_check.sample(frac=0.1)
df_check1.redemption_status.sum()
#Full dataset test & train

df_full = pd.concat([df_test,df_train],axis=0)

df_full = df_full.sample(frac=0.1)
df_full = df_full.set_index('campaign_id').join(df_camp.set_index('campaign_id')).reset_index()

#df_train = df_train.set_index('campaign_id').join(df_camp.set_index('campaign_id'),lsuffix='ca').reset_index()#
df_full = df_full.set_index('customer_id').join(df_demog.set_index('customer_id')).reset_index()
df_full = df_full.set_index('customer_id').join(df_ctran.set_index('customer_id')).reset_index()
df_full.size
df_full.isnull().sum()
df_full.dropna(inplace=True)
df_full.redemption_status.sum()
df_full.size
#df_full.to_csv('OutputData.csv',index=False)
df_full = df_full.set_index('item_id').join(df_item.set_index('item_id')).reset_index()
feature_cols = ['age_range', 'marital_status', 'rented', 'family_size',

       'no_of_children', 'income_bracket', 'campaign_type','quantity','selling_price','other_discount',

                 'coupon_discount','brand','brand_type','category']

impute_feature_cat= ['age_range','income_bracket', 'campaign_type', 'marital_status', 'rented','family_size']

lable = 'redemption_status'
encode_columns = ['age_range','marital_status','campaign_type','no_of_children']

encode_columns_2 = ['brand','brand_type','category']

encode_columns_3 = ['family_size']
df_full = EncodeColumns(df_full,orde,encode_columns,False)
df_full = EncodeColumns(df_full,orde,encode_columns_2,False)
df_full = EncodeColumns(df_full,orde,encode_columns_3,False)
#Impute columns

simple_imputer_cat = SimpleImputer(strategy='most_frequent')
#df_full.head()
trainData = df_full[df_full.data_type == 'train'].copy()
#trainData.groupby('redemption_status').size()
testData = df_full[df_full.data_type == 'test'].copy()

testData = testData.drop(columns =['redemption_status']).reset_index()
x_train,x_valid,y_train,y_valid = train_test_split(trainData[feature_cols],trainData[lable],test_size=0.10,random_state=rndval)
model = DecisionTreeClassifier(criterion='entropy',splitter='best')

clf = model.fit(x_train,y_train)

clf.score(x_train,y_train)

#plt.figure(figsize=[12,8])

#plot_tree(clf)
df_demog.head()