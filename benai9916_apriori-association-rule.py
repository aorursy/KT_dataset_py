import pandas as pd
import numpy as np
import io

import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
df = pd.read_csv('/kaggle/input/transactions-from-a-bakery/BreadBasket_DMS.csv')
# first five row
df.head()
# size of datset
df.shape
# summary about dataset
df.info()
# statistical summary of numerical variables
df.describe()
# check for missing values
df.isnull().sum() 
# merge date and time column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df[["Datetime", "Transaction", "Item"]]

df.head()
df.dtypes
# check for unique value in items
df['Item'].value_counts().to_dict()
# Remove none
df = df[df['Item'] != 'NONE']
# check NONE value removed or not
df[df['Item'] == 'NONE']
# Extract hour of the day and weekday of the week
# For Datetime: the day of the week are Monday=0, Sunday=6, thereby +1 to become Monday=1, Sunday=7

df['Hour'] = df['Datetime'].dt.hour

df["Weekday"] = df["Datetime"].dt.weekday + 1

df.head()
total_items = len(df)
total_days = len(np.unique(df.Datetime.dt.day))
total_months = len(np.unique(df.Datetime.dt.month))
average_items = int(total_items / total_days)
unique_items = df.Item.unique().size

print("Total unique_items: {} sold by the Bakery".format(unique_items))
print('-----------------------------')
print("Total sales: {} items sold in {} days throughout {} months".format(total_items, total_days, total_months))
print('-----------------------------')
print("Average_items daily sales: {}".format(average_items))
# Rank the top 10 best-selling items
counts = df.Item.value_counts()

percent = df.Item.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

top_10 = pd.DataFrame({'counts': counts, '%': percent})

top_10.head(10)
# Rank by percentage
plt.figure(figsize=(8,5))
df.Item.value_counts(normalize=True)[:10].plot(kind="bar", title="Percentage of Sales by Item").set(xlabel="Item", ylabel="Percentage")
plt.show()

# Rank by value
plt.figure(figsize=(8,5))
df.Item.value_counts()[:10].plot(kind="bar", title="Total Number of Sales by Item").set(xlabel="Item", ylabel="Total Number")
plt.show()
# set datetime as index 
df.set_index('Datetime', inplace=True)
# Number of items sold by day
df["Item"].resample("D").count().plot(figsize=(15,5), title="Total Number of Items Sold by Date").set(xlabel="Date", ylabel="Total Number of Items Sold")
plt.show()
# Number of items sold by month
df["Item"].resample("M").count().plot(figsize=(15,5), grid=True, title="Total Number by Items Sold by Month").set(xlabel="Date", ylabel="Total Number of Items Sold")
plt.show()
# Aggregate item sold by hour
df_groupby_hour = df.groupby("Hour").agg({"Item": lambda item: item.count()/total_days})
print(df_groupby_hour)

# Plot items sold by hour
plt.figure(figsize=(8,5))
sns.countplot(x='Hour',data=df)
plt.title('Items Sales by hour')
plt.show()
# sales groupby weekday
df_groupby_weekday = df.groupby("Weekday").agg({"Item": lambda item: item.count()})
df_groupby_weekday.head()
# Define dataset to machine learning
df_basket = df.groupby(["Transaction","Item"]).size().reset_index(name="Count")

market_basket = (df_basket.groupby(['Transaction', 'Item'])['Count'].sum().unstack().reset_index().fillna(0).set_index('Transaction'))
market_basket.head()
# Convert all of our numbers to either a 1 or a 0 (negative numbers are converted to zero, positive numbers are converted to 1)
def encode_data(datapoint):
  if datapoint <= 0:
    return 0
  else:
    return 1
# Process the transformation into the market_basket dataset
market_basket = market_basket.applymap(encode_data)

# Check the result
market_basket.head()

market_basket.isna().sum()
# Apriori method request a min_support: Support is defined as the percentage of time that an itemset appears in the dataset.
# Defined to start seeing data/results with min_support of 2%

itemsets = apriori(market_basket, min_support= 0.02, use_colnames=True)
# Build your association rules using the mxltend association_rules function.
# min_threshold can be thought of as the level of confidence percentage that you want to return
# Defined to use 50% of min_threshold

rules = association_rules(itemsets, metric='lift', min_threshold=0.5)
# Below the list of products sales combinations
# It can use this information to build a cross-sell recommendation system that promotes these products with each other 

rules.sort_values("lift", ascending = False, inplace = True)
rules.head(10)
support = rules.support.to_numpy()
confidence = rules.confidence.to_numpy()

for i in range (len(support)):
    support[i] = support[i]
    confidence[i] = confidence[i]

plt.figure(figsize=(8,6))    
plt.title('Assonciation Rules')
plt.xlabel('support')
plt.ylabel('confidance')
sns.regplot(x=support, y=confidence, fit_reg=False)
plt.show()
# Recommendation of Market Basket
rec_rules = rules[ (rules['lift'] > 1) & (rules['confidence'] >= 0.5) ]
# Recommendation of Market Basket Dataset
cols_keep = {'antecedents':'item_1', 'consequents':'item_2', 'support':'support', 'confidence':'confidence', 'lift':'lift'}
cols_drop = ['antecedent support', 'consequent support', 'leverage', 'conviction']

recommendation_basket = pd.DataFrame(rec_rules).rename(columns= cols_keep).drop(columns=cols_drop).sort_values(by=['lift'], ascending = False)

display(recommendation_basket)