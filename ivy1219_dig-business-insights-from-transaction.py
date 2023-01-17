import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_palette('Paired')
df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head(2)
df.Date = pd.to_datetime(df.Date)
df['Year'] = df.Date.dt.year.astype(int)
df['Month'] = df.Date.dt.month.astype(int)
df['Dow'] = df.Date.dt.dayofweek.astype(int)
df['Hour'] = df.Time.apply(lambda x: x.split(':')[0]).astype(int)

df['Morning'] = 0
df['Morning'].loc[df.Hour < 12] =1
df['Afternoon'] = 0
df['Afternoon'].loc[(df.Hour >= 12)&(df.Hour < 18)] =1
df['Evening'] = 0
df['Evening'].loc[df.Hour >= 18] =1
df.head(3)
       
df.Item.value_counts()[:10]
df= df.drop(df.loc[df['Item'] =='NONE'].index)
#df.Item.value_counts()[:10].plot(kind ='bar',figsize =(8,8),title = 'The top 10 popular goods')
values = df.Item.value_counts()[:10]
labels = df.Item.value_counts().index[:10]
plt.figure(figsize = (8,8))
plt.pie(values, autopct='%1.1f%%', labels = labels,
        startangle=90)
plt.title('Top 10 bestselling goods in Piechart')
df.Transaction.value_counts().plot(kind ='hist',figsize =(8,8),title = 'Distribution of Transaction')
f,axes= plt.subplots(4,2,figsize =(14,28))
sns.countplot(df.Year,ax =axes[0][0])
sns.barplot(df.Year,df.Transaction,ax =axes[0][1])

sns.countplot(df.Month,ax =axes[1][0])
sns.barplot(df.Month,df.Transaction,ax =axes[1][1])

sns.countplot(df.Dow,ax =axes[2][0])
sns.barplot(df.Dow,df.Transaction,ax =axes[2][1])

sns.countplot(df.Hour,ax =axes[3][0])
sns.barplot(df.Hour,df.Transaction,ax =axes[3][1])


cols =['Morning','Afternoon','Evening']
for c in cols:
   print('Number of Transaction in',c, df[c].sum() ,'.The percentage is', df[c].sum()/len(df))
Coffee = df[df['Item'] == 'Coffee']
Coffee_hour = Coffee.groupby('Hour').size().reset_index(name='Counts')

plt.figure(figsize=(12,8))
ax = sns.barplot(x='Hour', y = 'Counts', data = Coffee_hour)
ax.set(xlabel='hour of Day', ylabel='Coffee Sold')
ax.set_title('Distribution of Coffees Sold by Hour of Day')

Coffee_tran =Coffee.Transaction.tolist()
df_subset = df[df.Transaction.isin(Coffee_tran)]
df_subset.head()
Top10_in_df = df.Item.value_counts()[:10]
Top10_in_subset = df_subset.Item.value_counts()[:10]

Top10_in_df = Top10_in_df.drop(labels=['Coffee'])
Top10_in_subset = Top10_in_subset.drop(labels=['Coffee'])

labels = Top10_in_df.index.values.tolist()
values_Top10_in_df =Top10_in_df.tolist()
values_Top10_in_subset = Top10_in_subset.tolist()

values_without_coffee = [values[i]-v for i,v in enumerate(values_Top10_in_df)]
coffee_mate = pd.DataFrame({'Name':labels,'with_':values_Top10_in_df, 'without_':values_Top10_in_subset})
coffee_mate

coffee_mate['with on without Percentage'] = coffee_mate.with_ / coffee_mate.without_ 
sns.mpl.rc('figure', figsize=(9,6))
sns.barplot('Name','with on without Percentage',data = coffee_mate)
plt.axhline(y=1, color='r', linestyle='--')
plt.axhline(y =2,color ='r',linestyle ='--')
plt.title('The percentage of item sales with coffee on items without coffee')
Coffee_count = Coffee.groupby('Date').size().reset_index(name ='Count')
Coffee_count.describe()
sns.mpl.rc('figure', figsize=(16,6))
for xc in np.arange(0, len(Coffee_count), step=7):
    plt.axvline(x=xc, color='k', linestyle='--')
    
Coffee_count.Count.plot(color ='r')
plt.xticks(np.arange(0, len(Coffee_count), step=7))
plt.ylabel('Coffee Sale')
plt.title('Daily Coffee sales')

ls=[]
for item in df['Transaction'].unique():
    ls2=list(set(df[df['Transaction']==item]['Item']))
    if len(ls2)>0:
        ls.append(ls2)
print(ls[0:3],len(ls))

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
te=TransactionEncoder()
te_data=te.fit(ls).transform(ls)
data_x=pd.DataFrame(te_data,columns=te.columns_)
print(data_x.head())

frequent_items= apriori(data_x, use_colnames=True, min_support=0.02)
print(frequent_items.head())

rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
rules
fig, ax=plt.subplots(figsize=(10,6))
GA=nx.from_pandas_edgelist(rules,source='antecedents',target='consequents')
nx.draw(GA,with_labels=True)
