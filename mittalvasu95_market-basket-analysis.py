# importing libraries



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



from wordcloud import WordCloud, STOPWORDS



from mlxtend.frequent_patterns import association_rules, apriori



import warnings

warnings.filterwarnings('ignore')
# reading the data

data = pd.read_csv('/kaggle/input/the-bread-basket/bread basket.csv')



# looking top 10 rows

data.head(10)
# looking the bigger picture

data.info()
# Converting the 'date_time' column into the right format

data['date_time'] = pd.to_datetime(data['date_time'])
data.head()
# Count of unique customers

data.Transaction.nunique()
# Extracting date

data['date'] = data['date_time'].dt.date

data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')



# Extracting time

data['time'] = data['date_time'].dt.time



# Extracting month and replacing it with text

data['month'] = data['date_time'].dt.month

data['month'] = data['month'].replace((1,2,3,4,5,6,7,8,9,10,11,12), 

                                          ('January','February','March','April','May','June','July','August',

                                          'September','October','November','December'))



# Extracting hour

data['hour'] = data['date_time'].dt.hour

# Replacing hours with text

hour_in_num = (1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)

hour_in_obj = ('1-2','7-8','8-9','9-10','10-11','11-12','12-13','13-14','14-15',

               '15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24')

data['hour'] = data['hour'].replace(hour_in_num, hour_in_obj)



# Extracting weekday and replacing it with text

data['weekday'] = data['date_time'].dt.weekday

data['weekday'] = data['weekday'].replace((0,1,2,3,4,5,6), 

                                          ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))



# dropping date_time column

data.drop('date_time', axis = 1, inplace = True)



data.head()
# cleaning the item column

data['Item'] = data['Item'].str.strip()

data['Item'] = data['Item'].str.lower()
# looking 10 rows of data

data.head(10)
all_headlines = ' '.join(data['Item'])

wordcloud = WordCloud(width = 3000, height = 2000, background_color = 'white', 

                      collocations = False).generate((all_headlines))

plt.figure(figsize = (15, 5))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x = data.Item.value_counts().head(20).index, y = data.Item.value_counts().head(20).values, color='pink')

plt.xlabel('Items', size = 15)

plt.xticks(rotation=45)

plt.ylabel('Count of Items', size = 15)

plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)

plt.show()
monthTran = data.groupby('month')['Transaction'].count().reset_index()

monthTran.loc[:,"monthorder"] = [4,8,12,2,1,7,6,3,5,11,10,9]

monthTran.sort_values("monthorder",inplace=True)





plt.figure(figsize=(12,5))

sns.barplot(data = monthTran, x = "month", y = "Transaction")

plt.xlabel('Months', size = 15)

plt.ylabel('Orders per month', size = 15)

plt.title('Number of orders received each month', color = 'green', size = 20)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x = data.period_day.value_counts().index, y = data.period_day.value_counts().values, color='pink')

plt.xlabel('Period', size = 15)

plt.ylabel('Orders per period', size = 15)

plt.title('Number of orders received in each period of a day', color = 'green', size = 20)

plt.show()
hourTran = data.groupby('hour')['Transaction'].count().reset_index()

hourTran.loc[:,"hourorder"] = [1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,7,8,9]

hourTran.sort_values("hourorder",inplace=True)



plt.figure(figsize=(12,5))

sns.barplot(data = hourTran, x = "hour", y = "Transaction")

plt.xlabel('Hours', size = 15)

plt.ylabel('Orders each hour', size = 15)

plt.title('Count of orders received each hour', color = 'green', size = 20)

plt.show()
weekTran = data.groupby('weekday')['Transaction'].count().reset_index()

weekTran.loc[:,"weekorder"] = [5,1,6,7,4,2,3]

weekTran.sort_values("weekorder",inplace=True)





plt.figure(figsize=(10,5))

sns.barplot(data = weekTran, x = "weekday", y = "Transaction", color='pink')

plt.xlabel('Weekdays', size = 15)

plt.ylabel('Orders per weekday', size = 15)

plt.title('Number of orders received each of the weekday', color = 'green', size = 20)

plt.show()
data.groupby('date')['Transaction'].count().plot(kind="line",figsize=(15,7),color='purple')

plt.xlabel('Date', size = 15)

plt.ylabel('Count of Transaction', size = 15)

plt.hlines(y = 129, color='red', xmin=data['date'].min(), xmax=data['date'].max(),

           linestyles='dashed', label='Mean:129')

plt.title('Transactions per day',  color = 'green', size = 20)

plt.legend(fontsize='large')

plt.show()
# getting dates where number of transactions are more than 200

dates = data.groupby('date')['Transaction'].count().reset_index()

dates = dates[dates['Transaction']>=200].sort_values('date').reset_index(drop=True)



dates = pd.merge(dates,data[['date','weekday']],on='date', how='inner')

dates.drop_duplicates(inplace=True)

dates
df = data.groupby(['period_day','Item'])['Transaction'].count().reset_index().sort_values(['period_day','Transaction'],ascending=False)

day = ['morning','afternoon','evening','night']



plt.figure(figsize=(15,8))

for i,j in enumerate(day):

    plt.subplot(2,2,i+1)

    df1 = df[df.period_day==j].head(10)

    sns.barplot(data=df1, y=df1.Item, x=df1.Transaction, color='pink')

    plt.xlabel('')

    plt.ylabel('')

    plt.title('Top 10 items people like to order in "{}"'.format(j), size=13)



plt.show()
# grouping the data with respect to transaction and item and look at the count of each item in each transaction



df = data.groupby(['Transaction','Item'])['Item'].count().reset_index(name='Count')

df
# making a mxn matrice where m=transaction and n=items and each row represents whether the item was in the transaction or not



my_basket = df.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)

# df.groupby(['Transaction','Item'])['Count'].sum().unstack().reset_index().fillna(0).set_index('Transaction')



my_basket.head()
# making a function which returns 0 or 1

# 0 means item was not in that transaction, 1 means item present in that transaction



def encode(x):

    if x<=0:

        return 0

    if x>=1:

        return 1



# applying the function to the dataset



my_basket_sets = my_basket.applymap(encode)

my_basket_sets.head()
# using the 'apriori algorithm' with min_support=0.01 (1% of 9465)

# It means the item should be present in atleast 94 transaction out of 9465 transactions only when we considered that item in

# frequent itemset



frequent_itemsets = apriori(my_basket_sets, min_support = 0.01, use_colnames = True)

frequent_itemsets
# now making the rules from frequent itemset generated above



rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)

rules.sort_values('confidence', ascending = False, inplace = True)

rules
# arranging the data from highest to lowest with respect to 'confidence'



rules.sort_values('confidence', ascending=False)