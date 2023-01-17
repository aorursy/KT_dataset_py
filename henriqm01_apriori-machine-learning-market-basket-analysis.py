#pip install pandas mlxtend

#!pip install pandas-profiling

#!pip install --upgrade pandas_profiling
# Importation of libraries

import pandas as pd

import numpy as np

import io



import matplotlib.pyplot as plt

import seaborn as sns



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

import mlxtend as ml



print(f'Libraries loaded!')
# Open and Creating the dataset

df_initial = pd.read_csv('../input/transactions-from-a-bakery/BreadBasket_DMS.csv', header = 0)

df_initial.head()
# Dataset info

display(df_initial.info())



# Descriptive statistics of the data

display(df_initial.describe())
# Identify null values on dataset

print('  ' * 10 + " Display information about column types and number of null values " + '  ' * 10 )

print('--' * 50)



tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})

tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values'}))

tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.rename(index={0:'null values (%)'}))



if(any(df_initial.isnull().any())):

    print()

    display(tab_info)

else:

    print('NO missing data')
#Let's check the 'hidden' missing values in the dataset

missing_value = ["NaN", "NONE", "None", "Nil", "nan", "none", "nil", 0]



print('---------------------------------------------------------')

print("There are {0} 'hidden' missing values in the 'Item'column.".format(len(df_initial[df_initial.Item.isin(missing_value)])))

print("There are {0} 'hidden' missing values in the 'Transaction'.".format(len(df_initial[df_initial.Transaction.isin(missing_value)])))

print('---------------------------------------------------------')

df_initial[df_initial.Item.isin(missing_value)].head()
# Selecting the row values to drop in the selected column

bread = df_initial.drop(df_initial[df_initial.Item == "NONE"].index)

print("Number of rows: {0:,} (original 21,293) ".format(len(bread)))

print('----------------------------------------')

bread.head()



# After removing the missing values, the number of rows left is 20,507 (original 21,293 minus 786 missing)
# Creating a column of DateTimeIndex

bread['Datetime'] = pd.to_datetime(bread['Date'] + ' ' + bread['Time'])

bread = bread[["Datetime", "Transaction", "Item"]].set_index("Datetime")

bread.head()
# Extract hour of the day and weekday of the week

# For Datetimeindex: the day of the week are Monday=0, Sunday=6, thereby +1 to become Monday=1, Sunday=7

bread["Hour"] = bread.index.hour

bread["Weekday"] = bread.index.weekday + 1



bread.head()
total_items = len(bread)

total_days = len(np.unique(bread.index.date))

total_months = len(np.unique(bread.index.month))

average_items = int(total_items / total_days)

unique_items = bread.Item.unique().size



print("Total unique_items: {} sold by the Bakery".format(unique_items))

print('-----------------------------')

print("Total sales: {} items sold in {} days throughout {} months".format(total_items, total_days, total_months))

print('-----------------------------')

print("Average_items daily sales: {}".format(average_items))
# Rank the top 10 best-selling items

counts = bread.Item.value_counts()

percent = bread.Item.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

top_10 = pd.DataFrame({'counts': counts, '%': percent})[:10]



print('-----------------------------')

print('Top 10 items')

print('-----------------------------')

display(top_10)
# Rank by percentage

plt.figure(figsize=(8,5))

bread.Item.value_counts(normalize=True)[:10].plot(kind="bar", title="Percentage of Sales by Item").set(xlabel="Item", ylabel="Percentage")

plt.show()



# Rank by value

plt.figure(figsize=(8,5))

bread.Item.value_counts()[:10].plot(kind="bar", title="Total Number of Sales by Item").set(xlabel="Item", ylabel="Total Number")

plt.show()
# Number of items sold by day

bread["Item"].resample("D").count().plot(figsize=(15,5), grid=True, title="Total Number of Items Sold by Date").set(xlabel="Date", ylabel="Total Number of Items Sold")

plt.show()
# Number of items sold by month

bread["Item"].resample("M").count().plot(figsize=(15,5), grid=True, title="Total Number by Items Sold by Month").set(xlabel="Date", ylabel="Total Number of Items Sold")

plt.show()
# Aggregate item sold by hour

bread_groupby_hour = bread.groupby("Hour").agg({"Item": lambda item: item.count()/total_days})

print(bread_groupby_hour)



# Plot items sold by hour

plt.figure(figsize=(8,5))

sns.countplot(x='Hour',data=bread)

plt.title('Items Sales by hour')

plt.show()
# sales groupby weekday

bread_groupby_weekday = bread.groupby("Weekday").agg({"Item": lambda item: item.count()})

bread_groupby_weekday.head()
# but we need to find out how many each weekday in that period of transaction

# in order to calculate the average items per weekday



import datetime 

daterange = pd.date_range(datetime.date(2016, 10, 30), datetime.date(2017, 4, 9))



monday = 0

tuesday = 0

wednesday = 0

thursday = 0

friday = 0

saturday = 0

sunday = 0



for day in np.unique(bread.index.date):

    if day.isoweekday() == 1:

        monday += 1

    elif day.isoweekday() == 2:

        tuesday += 1

    elif day.isoweekday() == 3:

        wednesday += 1

    elif day.isoweekday() == 4:

        thursday += 1        

    elif day.isoweekday() == 5:

        friday += 1        

    elif day.isoweekday() == 6:

        saturday += 1        

    elif day.isoweekday() == 7:

        sunday += 1        

        

all_weekdays = monday + tuesday + wednesday + thursday + friday + saturday + sunday



print("monday = {0}, tuesday = {1}, wednesday = {2}, thursday = {3}, friday = {4}, saturday = {5}, sunday = {6}, total = {7}".format(monday, tuesday, wednesday, thursday, friday, saturday, sunday, all_weekdays))
# apply the conditions to calculate the average items for each weekday

conditions = [

    (bread_groupby_weekday.index == 1),

    (bread_groupby_weekday.index == 2),

    (bread_groupby_weekday.index == 3),

    (bread_groupby_weekday.index == 4),

    (bread_groupby_weekday.index == 5),

    (bread_groupby_weekday.index == 6),

    (bread_groupby_weekday.index == 7)]



choices = [bread_groupby_weekday.Item/21, bread_groupby_weekday.Item/23, bread_groupby_weekday.Item/23, bread_groupby_weekday.Item/23, bread_groupby_weekday.Item/23, bread_groupby_weekday.Item/23, bread_groupby_weekday.Item/23]



bread_groupby_weekday["Average"] = np.select(conditions, choices, default=0)

bread_groupby_weekday
bread_groupby_weekday.plot(y="Average", figsize=(12,5), title="Average Number by Items Sold by Day of the Week").set(xlabel="Day of the Week (1=Monday, 7=Sunday)", ylabel="Average Number of Items Sold")

plt.show()
# Define dataset to machine learning

df_basket = bread.groupby(["Transaction","Item"]).size().reset_index(name="Count")



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

recommendation_basket['item_1'] = recommendation_basket['item_1'].str.join('()')

recommendation_basket['item_2'] = recommendation_basket['item_2'].str.join('()')

display(recommendation_basket)