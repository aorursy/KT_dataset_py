# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')
#Reading the data
data = pd.read_csv("../input/BreadBasket_DMS.csv")
# Listing the data columns
data.info()
# Describing the quantitative column 
data.describe()
# look at head
data.head()
data.isnull().sum()
data['Date time']= pd.to_datetime(data['Date']+' '+data['Time'])
data['Date time'].dt.year.value_counts()
data['Year Month']=data['Date time'].map(lambda x: 100*x.year + x.month)
data['Hour']=data['Date time'].dt.hour
data['Day']=data['Date time'].dt.weekday_name
data['Weekend vs Weekday'] = data['Date time'].apply(lambda x: 'Weekend' if x.dayofweek//5==1 else 'Weekday')
plt.figure(figsize=[10,5])
plt.plot(data['Date time'], data['Transaction'])
plt.title('No of Transaction by DateTime')

Transaction_by_month=data[['Year Month','Transaction']].groupby('Year Month',as_index=False).sum()
plt.figure(figsize=[10,5])
sns.barplot(x='Year Month',y='Transaction',data=Transaction_by_month)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by month')
plt.figure(figsize=[10,5])
sns.boxplot(x='Day',y='Transaction',data=data)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Day')
plt.figure(figsize=[10,5])
sns.boxplot(x='Weekend vs Weekday',y='Transaction',data=data)
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Weekend vs Weekday')
plt.figure(figsize=[10,5])
plt.ticklabel_format(style='plain', axis='y')
plt.title('Sale by Hour')
plt.plot(data[['Hour','Transaction']].groupby('Hour').sum())
plt.figure(figsize=[10,5])
sns.distplot(data['Transaction'],bins=100)
Item_by_transaction=data[['Item','Transaction']].groupby('Item',as_index=False).sum().sort_values(by='Transaction',ascending=False)
Item_by_transaction['Transaction %']=Item_by_transaction['Transaction']/Item_by_transaction['Transaction'].sum()
plt.figure(figsize=[10,5])
sns.barplot(x='Item',y='Transaction',data=Item_by_transaction.head(10))
plt.ticklabel_format(style='plain', axis='y')
plt.title('Top 10 Items')
plt.xticks(rotation = 90)
plt.figure(figsize=[10,5])
sns.barplot(x='Item',y='Transaction %',data=Item_by_transaction.head(10))
plt.ticklabel_format(style='plain', axis='y')
plt.title('Top 10 Items')
plt.xticks(rotation = 90)
Hour_by_Item=data[['Hour','Item','Transaction']].groupby(['Hour','Item'],as_index=False).sum()
Top_items=list(Item_by_transaction['Item'].head(10))
plt.figure(figsize=[10,5])
sns.boxplot(x='Item',y='Transaction',data=data[data['Item'].isin(Top_items)])
plt.ticklabel_format(style='plain', axis='y')
plt.title('No of Transaction by Top 10 Item')
plt.figure(figsize=[13,5])
plt.ticklabel_format(style='plain', axis='y')
plt.title('Sale by Hour for Top 5 Items')
sns.lineplot(x='Hour',y='Transaction',data=Hour_by_Item[Hour_by_Item['Item'].isin(Top_items)],hue='Item')
Top25Items=list(Item_by_transaction['Item'].head(25))
dataTop25Items=data[data['Item'].isin(Top25Items)]
dataTop25Items_pivot = dataTop25Items[['Date','Item','Transaction']].pivot_table('Transaction', 'Date', 'Item')
# Correlation Plot
f, ax = plt.subplots(figsize=[12,10])
sns.heatmap(dataTop25Items_pivot.corr(),annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot")
plt.show()
dataTop25Items_unstack=dataTop25Items.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
dataTop25Items_unstack = dataTop25Items_unstack.applymap(lambda x: 0 if x<=0 else 1)
dataTop25Items_frequent = apriori(dataTop25Items_unstack, min_support=0.01, use_colnames=True)
assciation = association_rules(dataTop25Items_frequent, metric="lift", min_threshold=1)
assciation[assciation['confidence']>=0.4]
