# We will go with 3 parts

# part 1: data analysis

# part 2: Pareto analysis

# part 3: market basket analysis (this is my first time for this algorithm, please give your feedback, cheers)



# Importing all the necessary librabries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



#read data CSV file

raw_data=pd.read_csv('../input/BreadBasket_DMS.csv')
#PART 1- DATA ANALYSIS 
#removing all rows where item value is NONE

raw_data=raw_data[raw_data['Item']!='NONE']
#groupby item

no_of_item_transc= raw_data.groupby(['Item'])['Transaction'].count() 

no_of_item_transc=no_of_item_transc.reset_index()

no_of_item_transc=no_of_item_transc.sort_values(['Transaction'],ascending=False)

top_item_transc=no_of_item_transc.head(25)



#plot graph for highest sold top 25 items

fig,axis=plt.subplots(figsize=(10,6))

axis=sns.barplot(data=top_item_transc,x='Item',y='Transaction')

axis.set_xticklabels(top_item_transc['Item'],rotation=70,color='b')

axis.set_xlabel('Name Of The Items',color='red',fontsize=16)

axis.set_ylabel('Total # of Transcations of Items', fontsize=16,color='r')
#groupby month and year

month_year=raw_data.copy()  

month_year['Date']=pd.to_datetime(month_year['Date'])

month_year['Month'],month_year['Year']=month_year['Date'].dt.month,month_year['Date'].dt.year



grp_month_year=month_year.groupby(['Month','Year'])['Transaction'].count().reset_index()

grp_month_year['Period'] = grp_month_year.Month.astype(str).str.cat(grp_month_year.Year.astype(str), sep=',')



#plot graph for each month of the year 2016,2017

fig,axis=plt.subplots(figsize=(8,4))

axis=sns.barplot(data=grp_month_year,x='Period',y='Transaction')

axis.set_xlabel('Month & Year',fontsize=14,color='r')

axis.set_ylabel('# Of TRANSACTIONS',fontsize=14,color='r')

axis.set_xticklabels(grp_month_year['Period'],color='b',rotation=60)
#part of the day groupby

part_of_day=raw_data.copy()

part_of_day['timestamp'] = part_of_day.Date.astype(str).str.cat(part_of_day.Time.astype(str), sep=' ')

part_of_day['timestamp']=pd.to_datetime(part_of_day['timestamp'])



part_of_day['hour'] = part_of_day['timestamp'].dt.round('H').dt.hour

part_of_day.drop(['Time','Date'],axis=1,inplace=True)



#peak coffee hours

cofy_hours=part_of_day[part_of_day['Item']=='Coffee']

cofy_hours=cofy_hours.groupby('hour')['Item'].count()

cofy_hours=cofy_hours.reset_index()



fig,ax=plt.subplots(figsize=(10,6))

ax=sns.barplot(data=cofy_hours,x='hour',y='Item')

ax.set_xlabel('Hours Of The Day',fontsize=16,color='r')

ax.set_ylabel('No of Times Coffee is Sold',fontsize=16,color='r')
#peak Bread hours

bread_hours=part_of_day[part_of_day['Item']=='Bread']

bread_hours=bread_hours.groupby('hour')['Item'].count()

bread_hours=bread_hours.reset_index()



fig,ax=plt.subplots(figsize=(10,6))

ax=sns.barplot(data=bread_hours,x='hour',y='Item')

ax.set_xlabel('Hours Of The Day',fontsize=16,color='r')

ax.set_ylabel('No of Times Bread is Sold',fontsize=16,color='r')
#peak Cake hours

cake_hours=part_of_day.loc[(part_of_day['Item']=='Cake') | (part_of_day['Item']=='Pastry')]

cake_hours=cake_hours.groupby('hour')['Item'].count()

cake_hours=cake_hours.reset_index()



fig,ax=plt.subplots(figsize=(10,6))

ax=sns.barplot(data=cake_hours,x='hour',y='Item')

ax.set_xlabel('Hours Of The Day',fontsize=16,color='r')

ax.set_ylabel('No of Times Cake is Sold',fontsize=16,color='r')
#peak tea hours

tea_hours=part_of_day.loc[(part_of_day['Item']=='Tea')]

tea_hours=tea_hours.groupby('hour')['Item'].count()

tea_hours=tea_hours.reset_index()



fig,ax=plt.subplots(figsize=(9,6))

ax=sns.barplot(data=tea_hours,x='hour',y='Item')

ax.set_xlabel('Hours Of The Day',fontsize=16,color='r')

ax.set_ylabel('No of Times Tea is Sold',fontsize=16,color='r')
#PART 2 Pareto analysis
pareto_data=raw_data.copy()

total_transc=len(pareto_data)



eighty_percent_transc=(total_transc/100)*80

total_unique_items=pareto_data.Item.unique()

twenty_percent_top_items=(len(total_unique_items)/100)*20

total_transc_top_20_Items=no_of_item_transc.head(int(np.round(twenty_percent_top_items)))

total_transc_top_20_Items=total_transc_top_20_Items.Transaction.sum()



print('80% of Total Transcation is :', eighty_percent_transc,'\n' )

print('Total Transcation Of 20% Top Items is :', total_transc_top_20_Items,'\n' )

print ('So, we can say this almost complies to 80-20% pareto rule')
#PART 3, MACHINE LEARNING

#Market Basket Model
"""

Theory of Apriori Algorithm

There are three major components of Apriori algorithm:



Support

Confidence

Lift

We will explain these three concepts with the help of an example.



Suppose we have a record of 1 thousand customer transactions, and we want to find the Support, Confidence, and Lift for two items e.g. burgers and ketchup.

Out of one thousand transactions, 100 contain ketchup while 150 contain a burger. Out of 150 transactions where a burger is purchased, 50 transactions contain ketchup as well.

Using this data, we want to find the support, confidence, and lift.



Support

Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions.

Suppose we want to find support for item B. This can be calculated as:



Support(B) = (Transactions containing (B))/(Total Transactions)  

For instance if out of 1000 transactions, 100 transactions contain Ketchup then the support for item Ketchup can be calculated as:



Support(Ketchup) = (Transactions containingKetchup)/(Total Transactions)

Support(Ketchup) = 100/1000  

                 = 10%

                 

Confidence

Confidence refers to the likelihood that an item B is also bought if item A is bought. It can be calculated by finding the number of transactions where A and B are bought together,

divided by total number of transactions where A is bought. Mathematically, it can be represented as:



Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)  

Coming back to our problem, we had 50 transactions where Burger and Ketchup were bought together. While in 150 transactions, burgers are bought.

Then we can find likelihood of buying ketchup when a burger is bought can be represented as confidence of Burger -> Ketchup and can be mathematically written as:



Confidence(Burger→Ketchup) = (Transactions containing both (Burger and Ketchup))/(Transactions containing A)

Confidence(Burger→Ketchup) = 50/150  

                           = 33.3%

You may notice that this is similar to what you'd see in the Naive Bayes Algorithm, however, the two algorithms are meant for different types of problems.



Lift

Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold. Lift(A –> B) can be calculated by dividing Confidence(A -> B) divided by Support(B).

Mathematically it can be represented as:



Lift(A→B) = (Confidence (A→B))/(Support (B))  

Coming back to our Burger and Ketchup problem, the Lift(Burger -> Ketchup) can be calculated as:



Lift(Burger→Ketchup) = (Confidence (Burger→Ketchup))/(Support (Ketchup))

Lift(Burger→Ketchup) = 33.3/10  

                     = 3.33



Lift basically tells us that the likelihood of buying a Burger and Ketchup together is 3.33 times more than the likelihood of just buying the ketchup.

A Lift of 1 means there is no association between products A and B. Lift of greater than 1 means products A and B are more likely to be bought together.

Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.



"""
#from apyori import apriori 

from mlxtend.frequent_patterns import association_rules

from mlxtend.frequent_patterns import apriori



#preparing the data for applying apriori

# filling all the null vaues with zero and later unstacking them to fed to apriori to find CONFIDENCE, SUPPORT and LIFT

apr_data=raw_data.copy()



grp_apr_data = apr_data.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')

def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1

hot_encoded_df = grp_apr_data.applymap(encode_units)



# applying support with min value of 5%

frequent_itemsets =apriori(hot_encoded_df, min_support=0.05,use_colnames=True)

print('Items with min 5% support are:','\n', frequent_itemsets)
# applying lift to items which have min value of support of 5% 

lift_items = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)

print('most sold pair of items are:','\n',lift_items[['antecedents','consequents','consequent support','confidence','lift']])
"""

Please give your feedback and it encourages me

"""