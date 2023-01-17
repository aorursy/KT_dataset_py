import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import pandas.util.testing as tm
# reading offer details

offers = pd.read_csv('offers.csv.gz', compression='gzip')

print(offers.shape)

offers.head()
#checking how many unique values are there in each column

offers.nunique()
#Checking how the data looks in this table.

fig, axes = plt.subplots(2,2, figsize=(15,6))

offers.groupby('company')['offer'].count().sort_values().plot(kind='bar', title='Company vs. No. of Offers', grid=True, ax=axes[0,0])

offers.groupby('brand')['offer'].count().sort_values().plot(kind='bar', title='Brand vs. No. of Offers', grid=True, ax=axes[0,1])

offers.groupby('category')['offer'].count().sort_values().plot(kind='bar', title='Category vs. No. of Offers', grid=True, ax=axes[1,0])

offers.groupby('offer')['offervalue'].sum().sort_values().plot(kind='bar', title='Offer vs. Offer Value', grid=True, ax=axes[1,1])

plt.tight_layout()
# since a combination of company, brand and category uniquely identifies a product, 

# I want to see if product has many offers or just one.

offers.groupby(['company','brand','category'])['offer'].size().sort_values().plot(kind='bar', title='Product vs. No. of Offers', grid=True, figsize=(20,5))
#Checking how many offers need a minimum quantity of more than 2

print(offers.quantity.value_counts())
#this table shows that there are 10 products which have duplicate offers. I mean that the minimum purchase quantity

#required and. offer value is same for duplicate offers. THEY MAY BE COMBINED TOGETHER.

offers.groupby(['company','brand','category'])['offer','quantity','offervalue'].nunique()
#reading customer history data when the offer was given to them

trainHistory = pd.read_csv('trainHistory.csv.gz', compression='gzip')

trainHistory['offerdate'] = pd.to_datetime(trainHistory['offerdate'])

print(trainHistory.shape)

trainHistory.sort_values(by='repeattrips', ascending=False).head(1)
print('Ending date: ',trainHistory.offerdate.max())

print('Starting date: ',trainHistory.offerdate.min())

print('Offers were given in ', trainHistory.offerdate.max() - trainHistory.offerdate.min(), ' period')

#So the offers were given in a 60 days period from March 1 2013 to April 30 2013
#there are no null values in the dataset

trainHistory.isnull().sum()
#This shows that out of 37 offers in the offer table, there are 24 offers in trainHistory table.

trainHistory.nunique()
#Now I wanna check how many trips were made by each customer first. 

#This shows that the max trips are 2124 by a customer. That seem to be an outlier.

trainHistory.repeattrips.describe()
trainHistory[trainHistory.repeattrips > 100]

#This shows that there are 3 potential outliers for number of trips
#Let's see of the given customers, how may are repeaters

#Let's see how many customers are offered a particular offer

#How many chains are in each market

#How many customers repeated after given a particular offer

fig, axes = plt.subplots(2,2, figsize=(15,10))

trainHistory.groupby('offer')['id'].count().plot(kind='bar', title='Offers vs. Customer Count', grid=True, ax=axes[0,0])

trainHistory.repeater.value_counts().plot(kind='pie', title='% Repeat Customers', ax=axes[0,1], autopct='%1.1f%%', textprops=dict(color="black"))

trainHistory.groupby('market')['chain'].count().sort_values().plot(kind='bar', title='Chain Count in each Market', grid=True, ax=axes[1,0])

trainHistory.groupby(['offer','repeater']).size().unstack().plot(kind='bar', title='Offer vs. Repeaters', grid=True, ax=axes[1,1], stacked=True)

plt.tight_layout()
#merging trainHistory and Offers table to see how many customer repeated for a particular product (company, brand, category)

history = trainHistory.merge(offers, on='offer')

print(history.shape)

history.head(1)
fig, axes = plt.subplots(2,2, figsize=(15,10))

history.groupby(['company','repeater']).size().unstack().plot(kind='bar', title='Company vs. Repeaters', grid=True, stacked=True, ax=axes[0,0])

history.groupby(['brand','repeater']).size().unstack().plot(kind='bar', title='Brand vs. Repeaters', grid=True, stacked=True, ax=axes[0,1])

history.groupby(['category','repeater']).size().unstack().plot(kind='bar', title='Category vs. Repeaters', grid=True, stacked=True, ax=axes[1,0])

history.groupby(['company','brand','category','repeater']).size().unstack().plot(kind='bar', title='Product vs. Repeaters', grid=True, stacked=True, ax=axes[1,1])

plt.tight_layout()

#Companies ending with 7979, 9383, 0383 have higher % of repeat customers than others

#Brand 6732, 6926, 28840 have the higher % repeat customers than others

#Catogories 2119, 9909 have higher % of repeat customers than other

#When I combine all of them together to see exactly which product is getting higher repeat customers than average

#(7979,6732,9909), (9383,6926,2119) comes out to be the best ones. This shows is the same pattern as we saw

#individually in Company, Brand and Category 
fig, ax =plt.subplots(1, figsize=(20,5))

sns.boxplot(x = history.offer, y = history.repeattrips, palette="Set3")

plt.tight_layout()

#The 3 outliers are the same that we saw above. 

#So it would make sense to remove them from this plot and see how it looks
subset = history[history.repeater == 't']

fig, ax =plt.subplots(1, figsize=(20,5))

sns.boxplot(x = subset.offer, y = subset.repeattrips)

plt.ylim(-1, 15)

plt.tight_layout()

#This shows that out of those who are repeaters, Offer 044,052,329,501  have median value = 2, rest all have 1.
fig, ax =plt.subplots(0, figsize=(20,5))

subset.boxplot(column=['repeattrips'], by=['company','brand','category'], figsize=(20,10), rot=90, medianprops=dict(linestyle='-', linewidth=4))

plt.ylim(-1, 15)

plt.tight_layout(pad=5)

#This shows that out of all the unique products that are on offer, 4 of them ahave median repeat trips = 2,

#rest all of them have 1.

#Reading testHistory dataset where I have to predict if a person is going to be a repeat customer or not

testHistory = pd.read_csv('testHistory.csv.gz', compression='gzip')

testHistory.offerdate = pd.to_datetime(testHistory.offerdate)

print(testHistory.shape)

testHistory.head(3)
#I should now see how much overlap is there in id, chain, market and offer to see how much variation is there

#Check if there are any common customers in train and test data

columns = ['id','chain','market','offer']

for c in columns:

    train = trainHistory[c]

    test = testHistory[c]

    print('Common', c, len(set(train).intersection(set(test))))
print('Ending date: ',testHistory.offerdate.max())

print('Starting date: ',testHistory.offerdate.min())

print('Offers were given in ', testHistory.offerdate.max() - testHistory.offerdate.min(), ' period')

#So the offers were given in a 91 days period from May 1 2013 to July 31 2013
fig, axes = plt.subplots(2,2, figsize=(15,5))

trainHistory.groupby('offer')['id'].count().sort_values().plot(kind='bar', title='Training [Offers vs. Customer Count]', grid=True, ax=axes[0,0])

testHistory.groupby('offer')['id'].count().sort_values().plot(kind='bar', title='Test [Offers vs. Customer Count]', grid=True, ax=axes[0,1])

trainHistory.groupby('market')['chain'].count().sort_values().plot(kind='bar', title='Training [Chain Count in each Market]', grid=True, ax=axes[1,0])

testHistory.groupby('market')['chain'].count().sort_values().plot(kind='bar', title='Test [Chain Count in each Market]', grid=True, ax=axes[1,1])

plt.tight_layout()

# #Not much correlation between training & test dataset in terms of offers and markets
#Since the dataset is too large, I am interested to learn about only those products that are on offer based on the

#offer table



#To be able to do that, I need to find all products on offer in both training and test datasets.

#Merging testHistory table with offers

pred_history = testHistory.merge(offers, on='offer')

pred_history.head()
# The following commented code is to reduce the transactions dataset and get only those transactions where

# either of 'company', 'brand' or 'category' is on offer
# comp_tmp = history.company

# comp_tmp1 = pred_history.company

# comp_tmp = comp_tmp.append(comp_tmp1)

# comp_tmp = comp_tmp.unique()

# comp_tmp = set(comp_tmp)



# brand_tmp = history.brand

# brand_tmp1 = pred_history.brand

# brand_tmp = brand_tmp.append(brand_tmp1)

# brand_tmp = brand_tmp.unique()

# brand_tmp = set(brand_tmp)



# cat_tmp = history.category

# cat_tmp1 = pred_history.category

# cat_tmp = cat_tmp.append(cat_tmp1)

# cat_tmp = cat_tmp.unique()

# cat_tmp = set(cat_tmp)
#import modin.pandas as mpd
# transactions = pd.DataFrame(columns=['id','chain','dept','category','company','brand','date','productsize','productmeasure','purchasequantity','purchaseamount'])

# chunk_size = 2 * (10 ** 7)

# unique_set_tr = set()

# count = 0

# for chunk in mpd.read_csv('transactions.csv.gz', compression = 'gzip', chunksize = chunk_size,

#                         engine='c'):

#     internal_transactions = chunk.loc[(chunk['company'].isin(comp_tmp)) & 

#                                       (chunk['brand'].isin(brand_tmp)) & 

#                                       (chunk['category'].isin(cat_tmp))]

#     internal_transactions.to_csv('transactions_offer_'+str(count)+'.csv')

#     count = count + 1

#     print(count)
# import glob

# import pandas as pd

# extension = 'csv'

# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# offer_transactions = pd.concat([pd.read_csv(f) for f in all_filenames ])

# #export to csv

# offer_transactions.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
# offer_transactions = pd.read_csv('offer_transactions.csv')

offer_transactions = pd.read_csv('combined_csv.csv')

offer_transactions.date = pd.to_datetime(offer_transactions.date)

print(offer_transactions.shape)

offer_transactions.head(1)
print('Ending date: ', offer_transactions.date.max())

print('Starting date: ', offer_transactions.date.min())

print('Transactions are from ', offer_transactions.date.max() - offer_transactions.date.min(), ' period')

#So the transaction history is given in a 513 days period from March 2 2012 to July 28 2013
# Feature Engineering: To add meaningful data from reduced transactions table into the training & test dataset

#adding 9 features in PART A and 27 features in PART B, as explained in the presentation
# features = ['company','brand','category']

# for f in features:

#     #FEATURE TYPE 1

#     feature = pd.DataFrame(offer_transactions.groupby(['id',f]).size())

#     feature.columns = [f+'_count']

#     feature = feature.reset_index()

#     #adding the feature in training data

#     #No. of transactions done for each company, brand and category

#     history = pd.merge(history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     history.fillna({f+'_count': 0}, inplace=True)

#     #adding the featue in test data

#     pred_history = pd.merge(pred_history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     pred_history.fillna({f+'_count': 0}, inplace=True)

    

#     #FEATURE TYPE 2

#     #Total quantity purchased for each company, brand and category

#     feature = pd.DataFrame(offer_transactions.groupby(['id',f])['purchasequantity'].sum())

#     feature = feature.rename({'purchasequantity': f+'_purchasequantity'}, axis=1)

#     feature = feature.reset_index()

#     #adding the feature in training data

#     history = pd.merge(history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     history.fillna({f+'_purchasequantity': 0}, inplace=True)

#     #adding the featue in test data

#     pred_history = pd.merge(pred_history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     pred_history.fillna({f+'_purchasequantity': 0}, inplace=True)

    

#     #FEATURE TYPE 3

#     #Total amount purchased for each company, brand and category

#     feature = pd.DataFrame(offer_transactions.groupby(['id',f])['purchaseamount'].sum())

#     feature = feature.rename({'purchaseamount': f+'_purchaseamount'}, axis=1)

#     feature = feature.reset_index()

#     #adding the feature in training data

#     history = pd.merge(history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     history.fillna({f+'_purchaseamount': 0}, inplace=True)

#     #adding the featue in test data

#     pred_history = pd.merge(pred_history, feature, left_on=['id',f], right_on=['id',f], how='left')

#     pred_history.fillna({f+'_purchaseamount': 0}, inplace=True)
# #Adding FEATURE SET 4

# #How many times, quantity and amount was spent on a unique combination of company, brand & category

# feature = pd.DataFrame(offer_transactions.groupby(['id','company','brand','category']).size())

# feature.columns = ['product_count']

# feature = feature.reset_index()

# #adding the feature in training data

# history = pd.merge(history, feature, left_on=['id','company','brand','category'], right_on=['id','company','brand','category'], how='left')

# history.fillna({'product_count': 0}, inplace=True)

# #adding the featue in test data

# pred_history = pd.merge(pred_history, feature, left_on=['id','company','brand','category'], right_on=['id','company','brand','category'], how='left')

# pred_history.fillna({'product_count': 0}, inplace=True)
# features = ['purchasequantity','purchaseamount']

# for f in features:

#     feature = pd.DataFrame(offer_transactions.groupby(['id','company','brand','category'])[f].sum())

#     feature.columns = ['product_'+f]

#     feature = feature.reset_index()

#     #adding the feature in training data

#     #No. of transactions done for each company, brand and category

#     history = pd.merge(history, feature, left_on=['id','company','brand','category'], right_on=['id','company','brand','category'], how='left')

#     history.fillna({'product_'+f: 0}, inplace=True)

#     #adding the featue in test data

#     pred_history = pd.merge(pred_history, feature, left_on=['id','company','brand','category'], right_on=['id','company','brand','category'], how='left')

#     pred_history.fillna({'product_'+f: 0}, inplace=True)



# history.to_csv ('history_4_features.csv', index = False, header=True)

# pred_history.to_csv ('pred_history_4_features.csv', index = False, header=True)
#reading feature rich files

history = pd.read_csv('history_4_features.csv')

pred_history = pd.read_csv('pred_history_4_features.csv')

history.offerdate = pd.to_datetime(history.offerdate)

pred_history.offerdate = pd.to_datetime(pred_history.offerdate)
# from datetime import datetime, timedelta

# #Adding FEATURE SET 5 for training data

# #5.1 No. of transaction of customer for a company 7, 30, 60 days before receiving the offer 

# #5.2 No. of transaction of customer for a brand 7, 30, 60 days before receiving the offer

# #5.3 No. of transaction of customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = history.offerdate.unique()

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_cnt_'+str(timeframe)  

            

#             if newColumnName not in history.columns:

#                 history[newColumnName] = np.nan          

#             count_f = pd.DataFrame(iddf.groupby(['id', column]).size())

#             count_f = count_f.reset_index()

#             count_f = count_f.rename(columns={0: column+'_cnt_'+str(timeframe)})

#             count_f['offerdate'] = date

#             #adding features to training data

#             history = history.merge(count_f, how='left', on=['id', column, 'offerdate'])

#             history[newColumnName+'_x'] = history[newColumnName+'_x'].fillna(history[newColumnName+'_y'])

#             history = history.drop(newColumnName+'_y', axis=1)

#             history = history.rename(columns={newColumnName+'_x': newColumnName})

#             adding features to test data

        

# history.to_csv ('history_f5a.csv', index = False, header=True)

            
# from datetime import datetime, timedelta

# #Adding FEATURE SET 5 for test data

# #5.1 No. of transaction of customer for a company 7, 30, 60 days before receiving the offer 

# #5.2 No. of transaction of customer for a brand 7, 30, 60 days before receiving the offer

# #5.3 No. of transaction of customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = pred_history.offerdate.unique()

# count = 0

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_cnt_'+str(timeframe)  

            

#             if newColumnName not in pred_history.columns:

#                 pred_history[newColumnName] = np.nan          

#             count_f = pd.DataFrame(iddf.groupby(['id', column]).size())

#             count_f = count_f.reset_index()

#             count_f = count_f.rename(columns={0: column+'_cnt_'+str(timeframe)})

#             count_f['offerdate'] = date

#             #adding features to test data

#             pred_history = pred_history.merge(count_f, how='left', on=['id', column, 'offerdate'])    

#             pred_history[newColumnName+'_x'] = pred_history[newColumnName+'_x'].fillna(pred_history[newColumnName+'_y'])

#             pred_history = pred_history.drop(newColumnName+'_y', axis=1)

#             pred_history = pred_history.rename(columns={newColumnName+'_x': newColumnName})

#         count = count + 1

#         print(count)

        

# pred_history.to_csv ('pred_history_f5a.csv', index = False, header=True)

            
# from datetime import datetime, timedelta

# #Adding FEATURE SET 5 for training data

# #5.4 Total quantity purchased by a customer for a company 7, 30, 60 days before receiving the offer 

# #5.5 Total quantity purchased by a customer for a brand 7, 30, 60 days before receiving the offer

# #5.6 Total quantity purchased by a customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = history.offerdate.unique()

# # history1 = history.copy(deep=True)

# count = 0

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_qty_'+str(timeframe)  

#             if newColumnName not in history.columns:

#                 print('in if loop', newColumnName)

#                 history[newColumnName] = np.nan          

#             qty_f = pd.DataFrame(iddf.groupby(['id', column])['purchasequantity'].sum())

#             qty_f = qty_f.reset_index()

#             qty_f = qty_f.rename(columns={'purchasequantity': column+'_qty_'+str(timeframe)})

#             qty_f['offerdate'] = date

            

#             history = history.merge(qty_f, how='left', on=['id', column, 'offerdate'])

#             history[newColumnName+'_x'] = history[newColumnName+'_x'].fillna(history[newColumnName+'_y'])

#             history = history.drop(newColumnName+'_y', axis=1)

#             history = history.rename(columns={newColumnName+'_x': newColumnName})

           

        

#         count = count + 1

#         print(count)

        

# history.to_csv ('history_f5b.csv', index = False, header=True)
# from datetime import datetime, timedelta

# #Adding FEATURE SET 5

# #Adding FEATURE SET 5 for training data

# #5.4 Total quantity purchased by a customer for a company 7, 30, 60 days before receiving the offer 

# #5.5 Total quantity purchased by a customer for a brand 7, 30, 60 days before receiving the offer

# #5.6 Total quantity purchased by a customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = pred_history.offerdate.unique()

# # history1 = history.copy(deep=True)

# count = 0

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_qty_'+str(timeframe)  

#             if newColumnName not in pred_history.columns:

#                 print('in if loop', newColumnName)

#                 pred_history[newColumnName] = np.nan          

#             qty_f = pd.DataFrame(iddf.groupby(['id', column])['purchasequantity'].sum())

#             qty_f = qty_f.reset_index()

#             qty_f = qty_f.rename(columns={'purchasequantity': column+'_qty_'+str(timeframe)})

#             qty_f['offerdate'] = date

    

#             pred_history = pred_history.merge(qty_f, how='left', on=['id', column, 'offerdate'])

#             pred_history[newColumnName+'_x'] = pred_history[newColumnName+'_x'].fillna(pred_history[newColumnName+'_y'])

#             pred_history = pred_history.drop(newColumnName+'_y', axis=1)

#             pred_history = pred_history.rename(columns={newColumnName+'_x': newColumnName})

        

#         count = count + 1

#         print(count)

        

# pred_history.to_csv ('pred_history_f5b.csv', index = False, header=True)
# from datetime import datetime, timedelta

# #Adding FEATURE SET 5 for training data

# #5.7 Total amount spent by a customer for a company 7, 30, 60 days before receiving the offer 

# #5.8 Total amount spent by a customer for a brand 7, 30, 60 days before receiving the offer

# #5.9 Total amount spent by a customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = history.offerdate.unique()

# # history1 = history.copy(deep=True)

# count = 0

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_amt_'+str(timeframe)  

            

#             if newColumnName not in history.columns:

#                 history[newColumnName] = np.nan          

#             amt_f = pd.DataFrame(iddf.groupby(['id', column])['purchaseamount'].sum())

#             amt_f = amt_f.reset_index()

#             amt_f = amt_f.rename(columns={'purchaseamount': column+'_amt_'+str(timeframe)})

#             amt_f['offerdate'] = date

            

#             history = history.merge(amt_f, how='left', on=['id', column, 'offerdate'])

#             history[newColumnName+'_x'] = history[newColumnName+'_x'].fillna(history[newColumnName+'_y'])

#             history = history.drop(newColumnName+'_y', axis=1)

#             history = history.rename(columns={newColumnName+'_x': newColumnName})

            

#         count = count + 1

#         print(count)

        

# history.to_csv ('history_f5c.csv', index = False, header=True)
# #Adding FEATURE SET 5 for test data

# #5.7 Total amount spent by a customer for a company 7, 30, 60 days before receiving the offer 

# #5.8 Total amount spent by a customer for a brand 7, 30, 60 days before receiving the offer

# #5.9 Total amount spent by a customer for a category 7, 30, 60 days before receiving the offer

# columns = ['company','brand','category']

# timeframes = [7, 30, 60]

# dates = pred_history.offerdate.unique()

# # history1 = history.copy(deep=True)

# count = 0

# dataframeMap = {}

# for date in dates:

#     date = pd.to_datetime(date)

#     for timeframe in timeframes:

#         iddf = offer_transactions.loc[(offer_transactions.date < date) & 

#                                       (offer_transactions.date >= date - timedelta(days=timeframe))]

#         for column in columns:

#             newColumnName = column+'_amt_'+str(timeframe)  

            

#             if newColumnName not in pred_history.columns:

#                 pred_history[newColumnName] = np.nan          

#             amt_f = pd.DataFrame(iddf.groupby(['id', column])['purchaseamount'].sum())

#             amt_f = amt_f.reset_index()

#             amt_f = amt_f.rename(columns={'purchaseamount': column+'_amt_'+str(timeframe)})

#             amt_f['offerdate'] = date

            

#             pred_history = pred_history.merge(amt_f, how='left', on=['id', column, 'offerdate'])

#             pred_history[newColumnName+'_x'] = pred_history[newColumnName+'_x'].fillna(pred_history[newColumnName+'_y'])

#             pred_history = pred_history.drop(newColumnName+'_y', axis=1)

#             pred_history = pred_history.rename(columns={newColumnName+'_x': newColumnName})

            

#         count = count + 1

#         print(count)

        

# pred_history.to_csv ('pred_history_f5c.csv', index = False, header=True)
#Reading files with all the features

history = pd.read_csv('history_f5c.csv')

pred_history = pd.read_csv('pred_history_f5c.csv')

history = history.fillna(0)

pred_history = pred_history.fillna(0)
#Creating DUMMY VARIABLES for market, offer, chain, company, brand, category

dummies = ['market', 'offer', 'chain', 'company', 'brand', 'category']

for d in dummies:

    #adding dummies to train data

    dummy = pd.get_dummies(history[d], prefix=d, prefix_sep='_', drop_first=True)

    history = history.merge(dummy, left_index=True, right_index=True)

    #adding dummies to test data

    dummy = pd.get_dummies(pred_history[d], prefix=d, prefix_sep='_', drop_first=True)

    pred_history = pred_history.merge(dummy, left_index=True, right_index=True)



pred_history_id = pd.DataFrame(pred_history.id)



#deleting columns that are not required in training & test dataset

columns = ['id','chain','offer','market','prod_id','offerdate','quantity','category','brand','company']

for x in columns:

    if x in history.columns:

        history = history.drop([x], axis = 1)

    if x in pred_history.columns:

        pred_history = pred_history.drop([x], axis = 1)
#Now adding those columns to trainHistory & pred_history table which are not common to both

common_cols = set(history.columns) & set(pred_history)

print('Common column length: ', len(common_cols))

print('trainHistory column length: ', len(history.columns))

print('testHistory column length: ', len(pred_history.columns))

to_add_in_train = set(pred_history.columns) - common_cols

print('Columns to be added in trainHistory: ', to_add_in_train)

print('Columns to be added in trainHistory: ', len(to_add_in_train))

to_add_in_test = set(history.columns) - common_cols

print('Columns to be added in testHistory: ', to_add_in_test)

print('Columns to be added in testHistory: ', len(to_add_in_test))
for column in to_add_in_train:

    history[column] = 0

print('History length: ', len(history.columns))

for column in to_add_in_test:

    pred_history[column] = 0

print('pred_history length: ', len(pred_history.columns))

print('Common columns: ', len(set(history.columns) & set(pred_history.columns)))
history.repeater.replace(['t','f'],[1,0],inplace=True)

history.repeater = history.repeater.astype('int')

X = history.drop(['repeater', 'repeattrips'], axis = 1)

y = pd.DataFrame(history.repeater)

X.head()
#Data Modeling
#First thing is to split the data into training and testing, to make sure we don't overfit the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=history.repeater)

print('Shape of training data: ', X_train.shape, y_train.shape)

X_train.head()
print('Shape of test data: ', X_test.shape, y_test.shape)

X_test.head()
#Running Logistic Regression

model = LogisticRegression(max_iter=100000)

model.fit(X_train, y_train)
predicted_y = model.predict(X_train)

print('Training Results')

print('Accuracy Score: ', accuracy_score(y_train, predicted_y))

print('Classification Report:')

print(classification_report(y_train, predicted_y))

print('Confusion Matrix:')

confusion_matrix(y_train, predicted_y)
predicted_y = model.predict(X_test)

print('Validation Results')

print('Accuracy Score: ', accuracy_score(y_test, predicted_y))

print('Classification Report:')

print(classification_report(y_test, predicted_y))

print('Confusion Matrix:')

confusion_matrix(y_test, predicted_y)
#Spot Check analysis

import warnings

warnings.filterwarnings("ignore")



models = []

models.append(('LogisticRegression', LogisticRegression()))

models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))

models.append(('KNeighborsClassifier', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

models.append(('GaussianNB', GaussianNB()))

models.append(('AdaBoostClassifier', ensemble.AdaBoostClassifier()))

models.append(('BaggingClassifier', ensemble.BaggingClassifier()))

models.append(('ExtraTreesClassifier', ensemble.ExtraTreesClassifier()))

models.append(('GradientBoostingClassifier', ensemble.GradientBoostingClassifier()))

models.append(('RandomForestClassifier', ensemble.RandomForestClassifier()))

models.append(('PassiveAggressiveClassifier', linear_model.PassiveAggressiveClassifier()))

models.append(('RidgeClassifierCV', linear_model.RidgeClassifierCV()))

models.append(('SGDClassifier', linear_model.SGDClassifier()))

models.append(('Perceptron', linear_model.Perceptron()))

models.append(('BernoulliNB', naive_bayes.BernoulliNB()))

models.append(('GaussianNB', naive_bayes.GaussianNB())) 



seed = 5

results = []

names = []



# store predictions

from sklearn.model_selection import cross_val_predict

for name, model in models:

    kfold = KFold(n_splits=3, random_state=seed, shuffle=True)

    # store the metrics

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    message = "%s %s = %f %s = (%f)" % (name, 'Mean', cv_results.mean(), 'Standard Deviation', cv_results.std())

    print(message)
#Model improvement using hyper tunning parameters

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

# Set grid search params

max_depth = [3, 5, 7]

learning_rate = [0.1, 0.15]

random_state = [10, 20, 30]



grid_params_GRA = [{ 

    'max_depth': max_depth,

    'learning_rate': learning_rate,

    'random_state': random_state

    }]



clf = make_pipeline(StandardScaler(), 

                    GridSearchCV(ensemble.GradientBoostingClassifier(),

                                 param_grid=grid_params_GRA,

                                 cv=2,

                                 refit=True))



clf.fit(X_train, y_train)

predicted_y = clf.predict(X_test)
print('Validation Results')

print('Accuracy Score: ', accuracy_score(y_test, predicted_y))

print('Classification Report:')

print(classification_report(y_test, predicted_y))

print('Confusion Matrix:')

confusion_matrix(y_test, predicted_y)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

GBR_roc_auc = roc_auc_score(y_test, clf.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Gradient Boost Classifier (area = %0.2f)' % GBR_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Gradient Boosting Classifier ROC Curve')

plt.legend(loc="lower right")

plt.show()
pred_history = pred_history.drop(['repeater','repeattrips'], axis=1)

predictions = clf.predict(pred_history)

predictions = pd.DataFrame(predictions)

predictions = predictions.rename({0: 'repeatProbability'}, axis=1)

predictions.head()
predictions.repeatProbability.value_counts()
pred_history_id = pd.DataFrame(pred_history_id)

finalPredictions = pd.merge(pred_history_id, predictions, left_index=True, right_index=True)

finalPredictions.head()
finalPredictions.to_csv ('finalPredictions.csv', index = False, header=True)