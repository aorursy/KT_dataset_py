import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('../input/bakery-sales-data-2006-19/bakery_sales.csv')
df.head()
del df['Unnamed: 0']
df['Date'] = pd.to_datetime(df['Date'])
cols = ['cakes', 'pies', 'cookies', 'smoothies', 'coffee']
col_sum = df[cols].sum(axis=0).sort_values(ascending=False)

df1 = pd.DataFrame(col_sum, columns=['values'])
df1.reset_index(inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=df1, x='index', y='values')
plt.xlabel('Items', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.title('The most sold food item', fontsize=20)
plt.show()

df1= df.groupby(['promotion'])[cols].agg('sum')

f, ax = plt.subplots(figsize=(8, 6), dpi=100)
df1.plot(kind='bar', stacked=False, ax=ax)
plt.xticks(rotation='horizontal')
plt.xlabel('Promotion', fontsize=16)
plt.ylabel('Total Sale', fontsize=16)
plt.title('The effect of Promotion on Sales of Items', fontsize=16)
plt.show()
wk_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

df2 = df.groupby('weekday')[cols].agg('sum').reindex(labels=wk_order)

f, ax = plt.subplots(figsize=(10, 6), dpi=100)
df2.plot(kind='bar', stacked=False, ax=ax, )
plt.xticks(rotation='horizontal')
plt.xlabel('Weekdays', fontsize=16)
plt.ylabel('Total Sale', fontsize=16)
plt.title('Weekday Sales and Weekend Sales', fontsize=20)
plt.legend(title='Items', bbox_to_anchor=(1.2, 1))
plt.show()

df['year'] = df['Date'].dt.strftime('%Y')
df3 = df.groupby(['year'], as_index=True)[cols].agg('sum')

f, ax = plt.subplots(figsize=(10, 5), dpi=100)
df3[-2:].plot(kind='bar', stacked=False, ax=ax)
plt.xticks(rotation='horizontal')
plt.xlabel('Years', fontsize=16)
plt.ylabel('Total Sale', fontsize=16)
plt.title('Sales Pattern in last two years', fontsize=20)
plt.legend(title='Items', loc='upper right', bbox_to_anchor=(1.2, 1))
plt.show()

df['month'] = df['Date'].dt.strftime('%b')

df4 = df.groupby(['month'], as_index=False)[cols].agg('sum')
df4 = df4.reindex(index=[4, 3, 7, 0, 8, 6, 5, 1, 11, 10, 9, 2])


plt.subplots(figsize=(8, 12))
tidy = df4.melt(id_vars='month').rename(columns=str.title)
sns.barplot(data=tidy, y='Month', x='Value', hue='Variable')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Months', fontsize=16)
plt.ylabel('Total Sale', fontsize=16)
plt.title('Sales Pattern month wise', fontsize=20)
plt.legend(title='Items', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()