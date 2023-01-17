import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/BreadBasket_DMS.csv')
df['Item']=df['Item'].str.lower()
x=df['Item']== 'none'
print(x.value_counts())
df=df.drop(df[df.Item == 'none'].index)
len(df['Item'].unique())
df_for_top10_Items=df['Item'].value_counts().head(10)
Item_array= np.arange(len(df_for_top10_Items))
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
Items_name=['coffee','bread','tea','cake','pastry','sandwich','medialuna','hot chocolate','cookies','brownie']
plt.bar(Item_array,df_for_top10_Items.iloc[:])
plt.xticks(Item_array,Items_name)
plt.title('Top 5 most selling items')
plt.show()
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'],format= '%H:%M:%S' ).dt.hour
df['day_of_week'] = df['Date'].dt.weekday
d=df.loc[:,'Date']
weekday_names=[ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
Weekday_number=[0,1,2,3,4,5,6]
week_df = d.groupby(d.dt.weekday).count().reindex(Weekday_number)
Item_array_week= np.arange(len(week_df))

plt.figure(figsize=(15,5))
my_colors = 'rk'
plt.bar(Item_array_week,week_df, color=my_colors)
plt.xticks(Item_array_week,weekday_names)
plt.title('Number of Transactions made based on Weekdays')
plt.show()
dt=df.loc[:,'Time']
Hour_names=[ 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
time_df=dt.groupby(dt).count().reindex(Hour_names)
Item_array_hour= np.arange(len(time_df))
plt.figure(figsize=(15,5))
my_colors = 'rb'
plt.bar(Item_array_hour,time_df, color=my_colors)
plt.xticks(Item_array_hour,Hour_names)
plt.title('Number of Transactions made based on Hours')
plt.show()
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
hot_encoded_df=df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
hot_encoded_df = hot_encoded_df.applymap(encode_units)

frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(10)
rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]
support=rules.as_matrix(columns=['support'])
confidence=rules.as_matrix(columns=['confidence'])
import seaborn as sns
 
for i in range (len(support)):
    support[i] = support[i] 
    confidence[i] = confidence[i] 
     
plt.title('Association Rules')
plt.xlabel('support')
plt.ylabel('confidence')    
sns.regplot(x=support, y=confidence, fit_reg=False)
 