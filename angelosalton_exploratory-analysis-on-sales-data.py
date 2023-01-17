import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# get a sample for quick analysis
data = pd.read_csv('/kaggle/input/sales-data-for-a-chain-of-brazilian-stores/Sales Report.csv', sep=';', parse_dates=['Sale Date Time'])
data = data.sample(5000)
# optional: convert string types to category for faster analysis
#for col in data.select_dtypes('object').columns:
#    data[col] = data[col].astype('category')
data.head()
data.describe(include='all')
# create some new columns
data['Unit Price'] = data['Total']/data['Amount']
data['Markup'] = data['Unit Price']/data['Product Cost']*100

data['Year'] = data['Sale Date Time'].dt.year
data['Month'] = data['Sale Date Time'].dt.month
data['Week'] = data['Sale Date Time'].dt.week
data['Weekday'] = data['Sale Date Time'].dt.weekday
data['Hour'] = data['Sale Date Time'].dt.hour
data.dtypes
# top clients
top_clients = data.groupby('Client')['Client','Total'].sum().sort_values(by='Total', ascending=False).reset_index()
top_clients[:10]
data.groupby(['Product Category'])['Amount','Total'].sum().sort_values(by='Total', ascending=False)
# sales over time
data.groupby(['Year','Month','Product Category'])['Total'].sum().unstack().plot(figsize=(14,7), title='Sales over time')
# sales over time excluding fuels
data.loc[data['Product Category']!='Fuel'].groupby(['Year','Month','Product Category'])['Total'].sum().unstack().plot(figsize=(14,7), title='Sales over time, excluding fuels')
# markup over time
data.groupby(['Year','Month','Product Category'])['Markup'].mean().unstack().plot(figsize=(14,7), title='Markup over time, by category')
plt.ylim(100, 300) # adjust y-axis
# orders by time of day
data.groupby('Hour')['Order Number'].count().plot(kind='bar', width=1, figsize=(14,7), title='Orders by time of day')
# average ticket
data.groupby(['Year','Month','Week'])['Total'].mean().plot(title='Average ticket', figsize=(14,7))
data.groupby(['Year','Month','Form of payment'])['Total'].sum().unstack().plot(kind='area', title='Forms of payment over time', figsize=(14,7))