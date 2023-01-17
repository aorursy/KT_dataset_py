import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')

df.head()
df.info()
def get_info(df):

    print('number of variables: ',df.shape[1])

    print('number of cases: ',df.shape[0])

    print('-'*10)

    print('variables(columns) names: ')

    print(df.columns)

    print('-'*10)

    print('data-type of each variable: ')

    print(df.dtypes)

    print('-'*10)

    print('missing rows in columns: ')

    c=df.isnull().sum()

    print(c[c>0])

get_info(df)
df.isnull()
df.isnull().sum()
df.notnull().sum()
cleaned_df=df.drop('county',axis=1)
cleaned_df.dropna().head()
for i in cleaned_df.drop(['model','manufacturer','paint_color'],axis=1).columns:

    if cleaned_df[i].dtype=='float':

        cleaned_df[i]=cleaned_df[i].fillna(cleaned_df[i].mean())

    if cleaned_df[i].dtype=='object':

        cleaned_df[i]=cleaned_df[i].fillna(cleaned_df[i].mode()[0])

cleaned_df['year']=cleaned_df['year'].fillna(cleaned_df['year'].mode()[0])

cleaned_df['model']=cleaned_df['model'].fillna('Unknown')

cleaned_df['manufacturer']=cleaned_df['manufacturer'].fillna('Unknown')

cleaned_df['paint_color']=cleaned_df['paint_color'].fillna('Unknown')
cleaned_df.duplicated()
cleaned_df=cleaned_df.drop_duplicates(['id'])

print('done')
def odometer_status(val):

    if val>101729.96151504324:

        return  'alot'

    else:

        return 'little'

cleaned_df['odometer_status']=df['odometer'].apply(odometer_status)
cleaned_df[['odometer_status','odometer']].tail()
cleaned_df['price']=cleaned_df['price'].replace(0,cleaned_df['price'].median())
for i in cleaned_df.columns:

    changer=i.title()

    cleaned_df.rename(columns={i:changer},inplace=True)
cleaned_df.columns
import seaborn as sns

sns.boxplot('Price',data=cleaned_df)
price_stats=cleaned_df['Price'].describe()

price_stats
from scipy.stats import iqr

iqr=iqr(cleaned_df['Price'])

iqr
upper_bound=price_stats['75%']+(1.5*iqr)

lower_bound=price_stats['25%']-(1.5*iqr)

outliers={'above_upper':0,'below_lower':0}

indexes=[]

for i,j in enumerate(cleaned_df['Price'].values):

    if j>upper_bound :

        outliers['above_upper']+=1

        indexes.append(i)

    elif j<lower_bound:

        outliers['below_lower']+=1

        indexes.append(i)
outliers
drive_dummy_df=pd.get_dummies(cleaned_df['Drive'],prefix='Drive')

drive_dummy_df.tail(10)
cleaned_df['Cylinders']=cleaned_df['Cylinders'].str.replace('cylinders','')
cleaned_df['Cylinders'].head()
pd.merge(cleaned_df['Drive'],drive_dummy_df,right_index=True,left_index=True)
df1=pd.DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})

df2=pd.DataFrame({'key':['a','b','c'],'data2':range(3)})

pd.merge(df1,df2,on='key')
df3=pd.DataFrame({'key':['a','f','b'],'data3':range(3)})

pd.merge(df1,df3,how='right')
cleaned_df=pd.concat([cleaned_df,drive_dummy_df],axis=1)

cleaned_df=cleaned_df.drop('Drive',axis=1)
cleaned_df.columns