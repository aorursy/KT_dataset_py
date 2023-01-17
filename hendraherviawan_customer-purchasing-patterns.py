# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display, HTML



%matplotlib inline

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})

df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")

df.info() 
# Credit:https://www.kaggle.com/fabiendaniel/customer-segmentation



order_canceled = df['InvoiceNo'].apply(lambda x:int('C' in x))

n1 = order_canceled.sum()

n2 = df.shape[0]

print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))
# Item sale with Quantity <=0 or unitPrice < 0

print (((df['Quantity'] <= 0) | (df['UnitPrice'] < 0)).value_counts())



#Delete the negative values 

df = df.loc[(df['Quantity'] > 0) | (df['UnitPrice'] >= 0)]
%%timeit

df['yearmonth'] = df['InvoiceDate'].apply(lambda x: (100*x.year) + x.month)

df['Week'] = df['InvoiceDate'].apply(lambda x: x.strftime('%W'))

df['day'] = df['InvoiceDate'].apply(lambda x: x.strftime('%d'))

df['Weekday'] = df['InvoiceDate'].apply(lambda x: x.strftime('%w'))

df['hour'] = df['InvoiceDate'].apply(lambda x: x.strftime('%H'))
plt.figure(figsize=(12,6))

plt.title("Frequency of order by Month", fontsize=15)

InvoiceDate = df.groupby(['InvoiceNo'])['yearmonth'].unique()

InvoiceDate.value_counts().sort_index().plot.bar()
# 2010-12 Dataset end at

df.loc[df['yearmonth'] == 201012]['InvoiceDate'].max()
# 2011-12 Dataset end at

df.loc[df['yearmonth'] == 201112]['InvoiceDate'].max()
plt.figure(figsize=(12,6))

plt.title("Frequency of order by Week", fontsize=15)

InvoiceDate = df.groupby(['InvoiceNo'])['Week'].unique()

InvoiceDate.value_counts().sort_index().plot.bar()
plt.figure(figsize=(12,6))

plt.title("Frequency of order by Day", fontsize=15)

InvoiceDate = df.groupby(['InvoiceNo'])['day'].unique()

InvoiceDate.value_counts().sort_index().plot.bar()
grouped_df  = df.groupby(["day", "hour"])["InvoiceNo"].unique().reset_index()

grouped_df["InvoiceNo"] = grouped_df["InvoiceNo"].apply(len)



#grouped_df = df.groupby(["Weekday", "hour"])["InvoiceNo"].aggregate("count").reset_index()

grouped_df = grouped_df.pivot('day', 'hour', 'InvoiceNo')



plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Frequency of Day Vs Hour of day")

plt.show()
#[0] ~ Sunday

plt.figure(figsize=(12,6))

plt.title("Frequency of order by Weekday", fontsize=15)

InvoiceDate = df.groupby(['InvoiceNo'])['Weekday'].unique()

InvoiceDate.value_counts().sort_index().plot.bar()
plt.figure(figsize=(12,6))

plt.title("Frequency of order by hour of day", fontsize=15)

InvoiceDate = df.groupby(['InvoiceNo'])['hour'].unique()

(InvoiceDate.value_counts()).iloc[0:-1].sort_index().plot.bar()
grouped_df  = df.groupby(["Weekday", "hour"])["InvoiceNo"].unique().reset_index()

grouped_df["InvoiceNo"] = grouped_df["InvoiceNo"].apply(len)



#grouped_df = df.groupby(["Weekday", "hour"])["InvoiceNo"].aggregate("count").reset_index()

grouped_df = grouped_df.pivot('Weekday', 'hour', 'InvoiceNo')



plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Frequency of Day of week Vs Hour of day")

plt.show()
df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})

df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")



#remove the negative values and replace with nan

df.loc[df['Quantity'] <= 0, 'Quantity'] = np.nan

df.loc[df['UnitPrice'] < 0, 'UnitPrice'] = np.nan



df.dropna(inplace=True)



df['total_dollars'] = df['Quantity']*df['UnitPrice']



df['yearmonth'] = df['InvoiceDate'].apply(lambda x: (100*x.year) + x.month)

df['Week'] = df['InvoiceDate'].apply(lambda x: x.strftime('%W'))

df['day'] = df['InvoiceDate'].apply(lambda x: x.strftime('%d'))

df['Weekday'] = df['InvoiceDate'].apply(lambda x: x.strftime('%w'))

df['hour'] = df['InvoiceDate'].apply(lambda x: x.strftime('%H'))
#First Item Order

df_sort = df.sort_values(['CustomerID', 'StockCode', 'InvoiceDate'])

df_sort_shift1 = df_sort.shift(1)

df_sort_reorder = df_sort.copy()

df_sort_reorder['reorder'] = np.where(df_sort['StockCode'] == df_sort_shift1['StockCode'], 1,0)

df_sort_reorder.head(5)
#Top 10 Reorder item

pd.DataFrame((df_sort_reorder.groupby(['Description'])['reorder'].sum())).sort_values('reorder', ascending = False).head(10)
notreorder = (df_sort_reorder[df_sort_reorder['reorder'] == 0 ].groupby(['yearmonth'])['total_dollars'].sum())

reorder = (df_sort_reorder[df_sort_reorder['reorder'] == 1 ].groupby(['yearmonth'])['total_dollars'].sum())



yearmonth = pd.DataFrame([notreorder , reorder], index=['First Buy', 'Reorder']).transpose()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])
notreorder = (df_sort_reorder[df_sort_reorder['reorder'] == 0 ].groupby(['Week'])['total_dollars'].sum())

reorder = (df_sort_reorder[df_sort_reorder['reorder'] == 1 ].groupby(['Week'])['total_dollars'].sum())



yearmonth = pd.DataFrame([notreorder , reorder], index=['First Buy', 'Reorder']).transpose()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])
notreorder = (df_sort_reorder[df_sort_reorder['reorder'] == 0 ].groupby(['day'])['total_dollars'].sum())

reorder = (df_sort_reorder[df_sort_reorder['reorder'] == 1 ].groupby(['day'])['total_dollars'].sum())



yearmonth = pd.DataFrame([notreorder , reorder], index=['First Buy', 'Reorder']).transpose()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,9))



yearmonth.plot.bar(stacked=True, ax=axes[0])

yearmonth.plot.box(ax=axes[1])