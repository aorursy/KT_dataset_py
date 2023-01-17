import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head()
df.Item.value_counts()
df.isnull().any()
print("unique items :" ,df.Item.nunique())
df.Item.unique()
df.loc[(df['Item']=="NONE")].head() 
len(df.loc[(df['Item']=="NONE")])
df.drop(df[df['Item']=='NONE'].index, inplace=True)
df.info()
df.shape
df["date"] = pd.to_datetime(df['Date'])
df["dayname"] = df["date"].dt.day_name()
df.drop("date", axis=1, inplace = True)
df.head()
df["year"],df["month"],df["day"]=df["Date"].str.split('-').str
df["hour"],df["minute"],df["second"]=df["Time"].str.split(":").str
df.drop("Date", axis=1, inplace = True)
df.drop("Time", axis=1, inplace = True)
df.head()
df.info()
#season
df["month"]=df["month"].astype(int)
df.loc[(df['month']==12),'season']="winter"
df.loc[(df['month']>=1)&(df['month']<=3),'season']="winter"
df.loc[(df['month']>3)&(df['month']<=6),'season']="spring"
df.loc[(df['month']>6)&(df['month']<=9),'season']="summer"
df.loc[(df['month']>9)&(df['month']<=11),'season']="fall"

df.head()
plt.figure(figsize=(20,50))
df['Item'].value_counts().sort_values().plot.barh(title='Top Item Sales',grid=True)
plt.figure(figsize=(20,10))
sns.countplot(x='dayname',data=df).set_title('Pattern of Transcation Trend Throughout The Week',fontsize=25)
plt.figure(figsize=(15,5))
sns.countplot(x='season',data=df).set_title('Pattern of Transation Trend During Different Season\'s',fontsize=25)
df['season'].unique()
plt.figure(figsize=(15,5))
sns.countplot(x='month',data=df).set_title('Transation Trend During Month',fontsize=25)
plt.figure(figsize=(15,5))
sns.countplot(x='year',data=df).set_title('Transation Trend During Year',fontsize=25)
plt.figure(figsize=(15,5))
sns.countplot(df["day"].astype(int)).set_title('pattern of transcation for each day')
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(df["hour"].astype(int)).set_title('pattern of transcation for each hour')
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(df["hour"].astype(int), df["Transaction"].value_counts()).set_title('transcation per hour')
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(df["year"].astype(int), df["Transaction"].value_counts()).set_title("Transcation through year")
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(df["season"], df["Transaction"].value_counts()).set_title('Transcation through season')
plt.show()
plt.figure(figsize=(15,5))
sns.barplot(df["dayname"], df["Transaction"].value_counts()).set_title('Transcations per day')
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(df["Transaction"].value_counts())
plt.show()

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Convert the units to 1 hot encoded values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1    
encoding = df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction').astype(int)
encoding.head()
encoding.tail()
encoding = encoding.applymap(encode_units)
frequent_itemsets = apriori(encoding, min_support=0.01, use_colnames=True)
frequent_itemsets.head(21)
# Create the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules
output = rules.sort_values(by=['confidence'], ascending=False)
output
rules[ (rules['lift'] >= 1) &
       (rules['confidence'] >= 0.5) ]