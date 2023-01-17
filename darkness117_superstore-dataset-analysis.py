import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

plt.rcParams['font.sans-serif'] = ['SimHei']

warnings.filterwarnings('ignore')      
df = pd.read_csv('../input/superstore-data/superstore_dataset2011-2015.csv', encoding='ISO-8859-1')



df.rename(columns=lambda x: x.replace(' ', '_').replace('-', '_'), inplace=True)

df.head()
df.dtypes
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])

df['Year'] = df["Order_Date"].dt.year

df['Month'] = df['Order_Date'].values.astype('datetime64[M]')

df.head()
df.isnull().sum()
df.drop(["Postal_Code"], axis=1, inplace=True)
df.describe()
df.duplicated().sum()
df1=df[['Order_Date','Sales','Profit','Year','Month']]

df1
df1.groupby('Year')['Month'].value_counts()
sales=df1.groupby(['Year','Month']).sum()

sales
year_2011 = sales.loc[(2011,slice(None)),:].reset_index()

year_2012 = sales.loc[(2013,slice(None)),:].reset_index()

year_2013 = sales.loc[(2013,slice(None)),:].reset_index()

year_2014 = sales.loc[(2014,slice(None)),:].reset_index()

year_2011
Profit=pd.concat([year_2011['Profit'],year_2012['Profit'],year_2013['Profit'],year_2014['Profit']],axis=1)

Profit
Profit.columns=['2011','2012','2013','2014']

Profit.index=['Jau','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

Profit
Sum=Profit.sum()

Sum.plot(kind='barh')
Market_Year_Sales = df.groupby(['Market', 'Year']).agg({'Sales':'sum'}).reset_index()

Market_Year_Sales.head()
sns.barplot(x='Market',y='Sales',hue='Year',data=Market_Year_Sales)

plt.title('Market_Sales')
productId_count = df.groupby('Product_ID').count()['Customer_ID'].sort_values(ascending=False)

productId_count.head(10)
productId_amount = df.groupby('Product_ID').sum()['Sales'].sort_values(ascending=False)

productId_amount.head(10)
productId_Profit= df.groupby('Product_ID').sum()['Profit'].sort_values(ascending=False)

productId_Profit.head(10)
df["Segment"].value_counts().plot(kind='pie',shadow=True)