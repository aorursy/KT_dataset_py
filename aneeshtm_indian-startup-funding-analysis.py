# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/startup-funding/startup_funding.csv')

df['City  Location'] = df['City  Location'].str.replace('NaN','')

df['City  Location'] = df['City  Location'].str.replace('xa0','')

df['City  Location'] = df['City  Location'].str.replace('xc2','')

df['City  Location'] = df['City  Location'].str.replace('\\\\','')

df['City  Location'] = df['City  Location'].str.replace('Bengaluru','Bangalore')

df['City  Location'] = df['City  Location'].str.replace('New Delhi','NCR')

df['City  Location'] = df['City  Location'].str.replace('Noida','NCR')

df['City  Location'] = df['City  Location'].str.replace('Gurgaon','NCR')

df['City  Location'] = df['City  Location'].str.replace('Gurugram','NCR')

df['City  Location'] = df['City  Location'].str.replace('Delhi','NCR')

Z = df['City  Location'].value_counts().head(10)
Z.plot(kind='barh')
df.rename(columns={'Investorsxe2x80x99 Name' : 'Investors Name'},inplace=True)

df.rename(columns={'Date ddmmyyyy' : 'Date'},inplace=True)

df['Year'] = pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.year

df['Month'] = pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.month

df.drop(columns = ['Sr No'],inplace=True)

df.info()
Y = df.groupby(['City  Location']).agg({'Amount in USD': 'count'}).sort_values(by=['Amount in USD'],ascending=False).head(10)

sns.set()

Y.plot(kind='bar')
df.groupby(['City  Location']).agg({'Amount in USD': 'count'}).sort_values(by=['Amount in USD'],ascending=False).head(10)

df.loc[df['Industry Vertical']=='ECommerce']='eCommerce'

df.loc[df['Industry Vertical']=='ECommerce']='eCommerce'

df.loc[df['Industry Vertical']=='E-commerce']='eCommerce'

df.loc[df['Industry Vertical']=='Ecommerce']='eCommerce'

df.loc[df['Industry Vertical']=='E-Commerce']='eCommerce'

df.loc[df['Industry Vertical']=='FinTech']='Fin-Tech'

df.loc[df['Industry Vertical']=='IT']='Technology'

X = df['Industry Vertical'].value_counts().head(10)

print(X)
sns.set()

X.plot(kind='barh')
amount_usd = {

    "\\xc2\\xa020,000,000": "20,000,000",

    "\\xc2\\xa016,200,000": "16,200,000",

    "\\xc2\\xa0N/A": "0",

    "\\xc2\\xa0685,000": "685,000",

    "\\xc2\\xa019,350,000": "19,350,000",

    "\\xc2\\xa05,000,000": "5,000,000",

    "\\xc2\\xa010,000,000":"10,000,000"

}

for i,v in amount_usd.items():

    #df['Amount in USD'][df['Amount in USD']==i] = v

    df.loc[df['Amount in USD'] == i,v] = v
df_2015 = df.loc[df.Year == 2015]

df_2016 = df[df.Year == 2016]

df_2017 = df[df.Year == 2017]

df_2018 = df[df.Year == 2018]

df_2019 = df[df.Year == 2019]
df_2015.Month.value_counts().plot(kind='bar', color='green', label='2015')

df_2016.Month.value_counts().plot(kind='bar', color='blue',  label='2016')

df_2017.Month.value_counts().plot(kind='bar', color='yellow', label='2017')

df_2018.Month.value_counts().plot(kind='bar', color='red', label='2018')

df_2019.Month.value_counts().plot(kind='bar', color='violet', label='2019')
df['InvestmentnType'].value_counts().head(10)
investment_type = {

    "Seed/ Angel Funding": "Angel Funding",

    "Seed / Angel Funding": "Angel Funding",

    "Seed/Angel Funding": "Angel Funding",

    "Angel Round": "Angel Funding",

    "Seed Funding Round": "Seed Funding",

    "Series A": "Seed Funding",

    "Series A":"Seed Funding",

    "Series B":"Seed Funding",

    "Series C":"Seed Funding",

    "Series B (Extension)":"Seed Funding",

    "Series D":"Seed Funding"

}

for i,v in investment_type.items():

    df['Investment_Type'] = df.loc[df['InvestmentnType']==i] = v