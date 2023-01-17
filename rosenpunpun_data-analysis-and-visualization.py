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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pandas as pd

df  = pd.read_csv('../input/productdemandforecasting/Historical Product Demand.csv', parse_dates=['Date'])
df.head()

print(df.dtypes)
df.dropna(inplace=True)
df.isnull().values.any()

df.loc[df['Date'] == 'NA'].head()
df.sort_values('Date')[1:15]
df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
df['Order_Demand'] = df['Order_Demand'].astype('int64')
df.sort_values('Date')[1:15]


df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df.head()
print(df.dtypes)
df2 = df[['Year', 'Warehouse', 'Order_Demand']].groupby(['Year', 'Warehouse'], as_index=False).count()
df2.head(10)
df2  = df2.pivot(index='Year', columns='Warehouse', values='Order_Demand')
df2

df2.index = df2.index.map(int) # let's change the index values of df2 to type integer for plotting
df2.plot(kind='area', stacked=False, figsize=(20, 10))

plt.title('Order_Demand Trend')
plt.ylabel('Number of Order_Demand')
plt.xlabel('Years')
plt.show()
df2['Total'] = df2.sum(axis=1)

df2

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink', 'red']
explode_list = [0.2, 0, 0, 0, 0, 0, 0.2] # ratio for each year with which to offset each wedge.

df2['Total'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            colors=colors_list,  # add custom colors
                            explode=explode_list 
                            )

# scale the title up by 12% to match pctdistance
plt.title('Order_Demand Trend [2011 - 2017]', y=1.12) 

plt.axis('equal') 

# add legend
plt.legend(labels=df2.index, loc='upper left') 

plt.show()
df2.describe()
df2[['Whse_A', 'Whse_C', 'Whse_J', 'Whse_S']].plot(kind='box', figsize=(8, 6))

plt.title('Order_Demand Trend [2011 - 2017]')
plt.ylabel('Number of Order_Demand')

plt.show()
df3 = df[['Date', 'Product_Category', 'Order_Demand']].groupby(['Date', 'Product_Category'], as_index=False).count()
df3.sort_values(by=['Order_Demand'], ascending=False).head()
df3['Date'] = pd.to_datetime(df3['Date'])
df3['Year'] = df3['Date'].dt.year

df3.head()

import datetime
format_str = '%Y-%m-%d' # The format
sns.relplot(x="Date", y="Order_Demand", hue="Product_Category", data=df3, height=5, aspect=5).set(xlim=(datetime.datetime.strptime('2011-01-01', format_str), datetime.datetime.strptime('2017-12-31', format_str)), ylim=(0, 650))