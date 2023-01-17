# linear algebra

import numpy as np



# data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 



# working with datetime data

from datetime import datetime, timedelta 



# data visualisation

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# special module to split datset:

from sklearn.model_selection import train_test_split



# creating and training tools 

from sklearn.ensemble import RandomForestRegressor



# model accuracy assessment tools

from sklearn import metrics 



# import ast to convert string representation of list to list

import ast



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# fix random seed trying to make experimental reproducibility

RANDOM_SEED = 42
# fix modules version

!pip freeze > requirements.txt
# df_train = pd.read_csv('main_task.csv')

# df_test = pd.read_csv('kaggle_task.csv')

# sample_submission = pd.read_csv('sample_submission.csv')



df_train = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')

df_test = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')

sample_submission = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/sample_submission.csv')
# combining the train and the test data in one dataset

df_train['sample'] = 1 # train mark

df_test['sample'] = 0 # test mark

df_test['Rating'] = 0 # initial value for target

data = df_test.append(df_train, sort=False).reset_index(drop=True)
data.info()
data.sample(5)
# parsing data from TripAdviser

# from bs4 import BeautifulSoup    

# import requests  



# using sleep and random avoiding ban from Tripadvisor

# import time

# import random



# string processing

# import re
# creating a list with Restaurant_id and URL



# data_new = data.drop(['City', 'Name', 'Cuisine Style', 'Price Range','Reviews',

#                   'ID_TA', 'Ranking', 'Number of Reviews', 'sample', 'Rating'], axis = 1)



# data_list = data_new.reset_index().values.tolist()
# def parsingTA(data_list):

#     '''grabbing data about restaurant from Tripadvisor'''

#     for restaurant in data_list:

#         url = 'https://www.tripadvisor.ru' + restaurant[2]

#         soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')

        

#         print(restaurant[0])

#         pattern = re.compile('(\d+)')

        

#         # finding photo amount

#         photo_count = soup.find(class_='details')

#         if photo_count == None: 

#             restaurant.append(0)

#         else:

            

#             restaurant.append(int(''.join(pattern.findall(str(photo_count)))))

        

#         # finding review count

#         review_count = soup.find(class_='reviewCount')

#         if review_count == None:

#             restaurant.append(0)

#         else:

#             restaurant.append(int(''.join(pattern.findall(str(review_count)))))

        

#         # finding verification

#         verification_badge = soup.find(class_='ui_icon verified-checkmark-fill restaurants-claimed-badge-ClaimedBadge__icon--JSEju')

#         if verification_badge != None:

#             verification = 1

#         else:

#             verification = 0

#         restaurant.append(verification)

        

#         # finding rank

#         rank = soup.find(class_='header_popularity popIndexValidation')

#         if rank == None:

#             restaurant.append(0)

#         else:

#             rank1 = rank.find('span')

#             restaurant.append(int(''.join(pattern.findall(str(rank1)))))

        

#         # finding restaurants amount in this city 

#         restaurants_amount = soup.find(class_='header_popularity popIndexValidation')

#         if restaurants_amount == None:

#             restaurant.append(0)

#         else:

#             restaurant.append(int(''.join(pattern.findall(str(restaurants_amount.contents[1])))))

        

#         # finding type rates

#         rates1 = soup.find(class_='choices')

#         if rates1 == None:

#             for i in range(5):

#                     restaurant.append(0)

#         else:

#             rates = rates1.find_all(class_='row_num is-shown-at-tablet')

#             if rates == []:

#                 for i in range(5):

#                     restaurant.append(0)

#             else:

#                 for rate in rates:

#                     restaurant.append(int(''.join(pattern.findall(str(rate)))))

          

          # avoiding ban from Tripadvisor

#         time.sleep(random.randint(1,10))

        

# parsingTA(data_list[])

# creating new dataframe and saving to scv



# new_df = pd.DataFrame(data_list[], columns = ['ID','Restaurant_id', 'URL_TA', 'photo', 'review_count_new',

#                                               'verification', 'rank', 'restaurants_amount', 'rate_5',

#                                               'rate_4', 'rate_3', 'rate_2', 'rate_1'])

# new_df.to_csv('TripAdviser_add_info.csv', index = False)
# additional_info = pd.read_csv('TripAdviser_add_info.csv')

additional_info = pd.read_csv("../input/tripadviser-add-ifo/TripAdviser_add_info.csv")

additional_info = additional_info.drop(['ID', 'Restaurant_id', 'URL_TA'], axis = 1)



# concatenation additional dataset with original dataset

data = pd.concat([data, additional_info], axis = 1)
data['mean_rank'] = data['rank']/data['City'].map(data.groupby(['City'])['rank'].max())
# DEGRADE, doesn't work



# data['mean_rate_5'] = data['rate_5']/data['City'].map(data.groupby(['City'])['rate_5'].max())

# data['mean_rate_4'] = data['rate_4']/data['City'].map(data.groupby(['City'])['rate_4'].max())

# data['mean_rate_3'] = data['rate_3']/data['City'].map(data.groupby(['City'])['rate_3'].max())

# data['mean_rate_2'] = data['rate_2']/data['City'].map(data.groupby(['City'])['rate_2'].max())

# data['mean_rate_1'] = data['rate_1']/data['City'].map(data.groupby(['City'])['rate_1'].max())
# # creating new column characterizing nan values in Number of Reviews

# # DEGRADE



data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# filling nan values, cause of clasterization and counting mean value



data['Number of Reviews'] = data['Number of Reviews'].fillna(0)
# trying to normalize ranking in each city

# clastering restraunts in each city, finding restraunt with max number of reviews and 

# normalizing number regarding max value



data['mean_number_of_reviews'] = data['Number of Reviews']/data['City'].map(data.groupby(['City'])['Number of Reviews'].max())
data['Price Range'].value_counts()
# creating dummies from price range values

# Degrade MAE



# price_dict = {'$':'low', '$$ - $$$':'average', '$$$$':'high'}

# data['price_range'] = data['Price Range'].map(price_dict)



# data = pd.get_dummies(data, columns=[ 'price_range',], dummy_na=True)
def new_price_range(row):

#Function returns numbers 0-3 according values from Price Range



    if str(row['Price Range']) == 'nan':

        return 0

    elif row['Price Range'] == '$':

        return 1

    elif row['Price Range'] == '$$ - $$$':

        return 2

    elif row['Price Range'] == '$$$$':

        return 3



# Creating new Series, values are filling with function new_price_range

price_range = data.apply(lambda row: new_price_range(row), axis=1)



# Counting new values to test revisions

#print(price_range.value_counts())



# Adding new column

data['price_range'] = price_range
# adding column wich shows nan value in Price Range



data['price_na'] = pd.isna(data['Price Range']).astype('uint8')

data['price_na'].value_counts()
# trying to normalize value in each city



data['mean_price_in_city'] = data['City'].map(data.groupby(['City'])['price_range'].mean())
city_list = data['City'].unique()



# creating list of capital cities

Capitals = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid',

            'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen','Amsterdam', 'Lisbon', 'Prague',

            'Oslo','Helsinki', 'Edinburgh', 'Ljubljana', 'Athens', 'Luxembourg']
def new_city(row):

    '''Function returns 1 if restaurant located in capital city '''

    if row['City'] in Capitals:

        return 1

    else:

        return 0





# Creating new Series, values are filling with function new_city

is_in_capital = data.apply(lambda row: new_city(row), axis=1)



# Counting new values to test revisions

#print(is_in_capital.value_counts())



# Adding new column if city is capital

data['is_in_capital'] = is_in_capital
# population, mln people

population = {'Paris': 2.141, 'Stockholm': 0.973, 'London': 8.9, 'Berlin': 3.748, 

              'Munich': 1.456, 'Oporto': 0.214,'Milan': 1.352,'Bratislava': 0.424, 

              'Vienna': 1.889, 'Rome': 2.873, 'Barcelona': 5.515, 'Madrid': 6.55,

              'Dublin': 1.361,'Brussels': 0.174, 'Zurich': 0.403, 'Warsaw': 1.708, 

              'Budapest': 1.75, 'Copenhagen': 0.602,'Amsterdam': 0.822,'Lyon': 0.513, 

              'Hamburg': 1.822,'Lisbon': 0.505, 'Prague': 1.319, 'Oslo': 0.673,

              'Helsinki': 0.632,'Edinburgh': 0.482,'Geneva': 0.495, 'Ljubljana': 0.28,

              'Athens': 0.664, 'Luxembourg': 0.602,'Krakow': 0.769}



# Adding new column - city population

data['Population'] = data['City'].map(population)
# adding new dummy columns with cities 



def find_city(cell):

    if city == cell:

        return 1

    return 0



for city in city_list:

    data[city] = data['City'].apply(find_city)
# creating new column characterizing nan values in Cuisine Style 

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
# finding mean value of cuisines amount in one restraunt



def fill_cuisine(s):

    '''Function returns 1 if Cuisine Style is empty or number of cuisines'''

    if str(s) == 'nan':

        return 1

    else:

        return(len(ast.literal_eval(s)))



# Addind count of cuisines

data['cuisine_amount'] = data['Cuisine Style'].apply(fill_cuisine)
# analyzing distribution 



data['cuisine_amount'].describe()
# clastering restraunts in each city, finding restraunt with max amount of cuisines and 

# normalizing cuisines amount regarding max value



data['norm_cuisine_amount'] = data['cuisine_amount']/data['City'].map(data.groupby(['City'])['cuisine_amount'].max())
# counting amount of restraunts with each cuisine type



def counting_keys(dictionary, key_word):

    '''if key_word is in dictionary, functions increment value by 1,

    in other case function create new item in dictionary '''

    if key_word in dictionary.keys():

        dictionary[key_word] += 1

    else:

        dictionary[key_word] = 1



cuisines_amount = {}



# filling values

for row in data['Cuisine Style']:

    if str(row) != 'nan':

        cuisine_list = ast.literal_eval(row)

        for cuisine in cuisine_list:

            counting_keys(cuisines_amount,cuisine)

# analyzing popular and rare cuisines

# creating dataframe from dictionary cuisines_amount



df_cuisines_amount = pd.DataFrame.from_dict(cuisines_amount, orient='index', columns = ['count'])



#df_cuisines_amount.sort_values(by = 'count',ascending = False)
df_cuisines_amount.describe()
# list of rare cuisines



df_rare_cuisines = df_cuisines_amount[df_cuisines_amount['count']<10].sort_values(by = 'count',ascending = False)

#df_rare_cuisines
# creating new columns wich characterized rare cuisins

def find_rare_cuisine(cell):

    if str(cell) == 'nan':

        return 0

    if item in ast.literal_eval(cell):

        return 1

    return 0



for item in df_rare_cuisines.index:

    data[item] = data['Cuisine Style'].apply(find_rare_cuisine)
# list of popular cuisines

df_popular_cuisine = df_cuisines_amount[df_cuisines_amount['count']>1000].sort_values(by = 'count',ascending = True)

#df_popular_cuisine
# creating new columns wich characterized poppular cuisins



def find_popular_cuisine(cell):

    if str(cell) == 'nan':

        return 0

    if item in ast.literal_eval(cell):

        return 1

    return 0



for item in df_popular_cuisine.index:

    data[item] = data['Cuisine Style'].apply(find_popular_cuisine)
# filling nan values 

data['Reviews'] = data['Reviews'].fillna('[[], []]')
# finding the newest review of reviews in restraunts dataframe

def fill_last_review(row):

    '''replacing nan values from text reviews cause of ast.literal_eval error

       Function returns 01-01-1900 if reviews are empty

       Function returns one date if it is only one review

       Function returns the biggest date if there is two reviews'''

    

    str_review = ast.literal_eval(str(row['Reviews']).replace('nan','0'))

    if str_review == [[], []]:

        return 'NaN'

    elif len(str_review[1]) == 1:

        return(datetime.strptime(str_review[1][0],'%m/%d/%Y'))

    else:

        first_review_time = datetime.strptime(str_review[1][0],'%m/%d/%Y')

        second_review_time = datetime.strptime(str_review[1][1],'%m/%d/%Y')

        #print(row['Restaurant_id'])

        if first_review_time < second_review_time:

            return second_review_time

        else:

            return first_review_time



last_review = data.apply(lambda row:fill_last_review(row), axis=1)

# finding the eldest review of reviews in restraunts dataframe



def fill_old_review(row):

    '''replacing nan values from text reviews cause of ast.literal_eval error

       Function returns 01-01-1900 if reviews are empty

       Function returns one date if it is only one review

       Function returns the earliest date if there is two reviews'''

    

    str_review = ast.literal_eval(str(row['Reviews'].replace('nan','0')))

    if str_review == [[], []]:

        return 'NaN'

    elif len(str_review[1]) == 1:

        return(datetime.strptime(str_review[1][0],'%m/%d/%Y'))

    else:

        first_review_time = datetime.strptime(str_review[1][0],'%m/%d/%Y')

        second_review_time = datetime.strptime(str_review[1][1],'%m/%d/%Y')

        #print(row['Restaurant_id'])

        if first_review_time < second_review_time:

            return first_review_time

        else:

            return second_review_time



        

old_review = data.apply(lambda row:fill_old_review(row), axis=1)
# Adding difference between two reviews

data['difference'] = (last_review - old_review).dt.days
# adding difference between current day and last review

data['passed_time'] = (datetime.today()- last_review).dt.days
# adding columns with nan values in 'difference' and 'passed_time' columns



data['difference_isNAN'] = pd.isna(data['difference']).astype('uint8')

data['passed_time_isNAN'] = pd.isna(data['passed_time']).astype('uint8')
data['Ranking'].describe()
# this normalization degrade MAE

# data['Ranking'] = data['Ranking']/data['Ranking'].mean()
# trying to normalize ranking in each city

# clastering restraunts in each city, finding max rated restraunts and normalizing ranks regarding max ranking



data['mean_ranking'] = data['Ranking']/data['City'].map(data.groupby(['City'])['Ranking'].max())



#data['mean_ranking'].value_counts()
# mining int numbers from 'ID_TA' and 'URL_TA', good correlation with ranking



data['ID_int'] = data['ID_TA'].apply(lambda s: int(s[1:]))

data['URL_int'] = data['URL_TA'].str.split('-').apply(lambda s: int(s[1][1:]))
# correlation heatmap

# uninformative due to the large number of columns



plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
data.columns
# dropping object columns



data = data.drop(['City', 'Cuisine Style', 'Price Range','Reviews',

                  'URL_TA', 'ID_TA', 'Restaurant_id'], axis = 1)
# filling nan values in 'difference' and 'passed_time'



data['difference'] = data['difference'].fillna(0)

data['passed_time'] = data['passed_time'].fillna(0)

# selecting the test part

train_data = data.query('sample == 1').drop(['sample'], axis=1)



y = train_data.Rating.values            # target

X = train_data.drop(['Rating'], axis=1)



# using train_test_split for splitting test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# checking size

train_data.shape, X.shape, X_train.shape, X_test.shape
# creating model, setting are not changed 

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# fitting the model on test dataset

model.fit(X_train, y_train)



# using the trained model to predict the rating of restaurants in a test dataset.

# The predicted values are saved to the variable y_pred

y_pred = model.predict(X_test)
# analyzing the result of prediction

y_pred
# rounding predict with pitch 0.5



y_pred_round = []

for item in y_pred:

    y_pred_round.append(round(item/0.5)*0.5)

y_pred_round_array = np.asarray(y_pred_round)
y_pred_round_array
# Compare the predicted values (y_pred) with real (y_test), and see how much they differ on average

# Mean Absolute Error (MAE) shows the average deviation of the predicted values from the real ones.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round_array))
# displaying the most important features for the model

plt.rcParams['figure.figsize'] = (10,15)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')
# # creating submission file 



# test_data = data.query('sample == 0').drop(['sample'], axis=1)

# test_data = test_data.drop(['Rating'], axis=1)



# predict_submission = model.predict(test_data)



# # rounding predict with pitch 0.5

# predict_submission_round = []

# for item in predict_submission:

#     predict_submission_round.append(round(item/0.5)*0.5)

# predict_submission_round_array = np.asarray(predict_submission_round)



# sample_submission['Rating'] = predict_submission_round_array

# sample_submission.to_csv('submission.csv', index=False)

# sample_submission.head()


# review_good_words = ['good','unique','delicious','best','amazing','excellent','nice','clean','lovely','relaxed','great',

#                      'heavenly','liked','fantastic','tasty','fresh','relaxing','perfect','hillarious','loved',

#                      'outstanding','favourite','not bad']

# review_bad_words = ['wasting','boring','avoid','overpriced','average','disappointing','standard','terrible','expensive',

#                    'shame','rude','slow','horrible','catastrophy','worst','aggressive','dirty','very bad','nothing']



# def fill_good_reviews(row):

#     str_review = ast.literal_eval(row.replace('nan','0'))

#     if len(str_review[0]) == 0:

#         return 0

#     elif len(str_review[0]) > 0:

#         for word in review_good_words:

#             if str(str_review[0][0]).find(word) != -1:

#                 return 1

#     elif len(str_review[0]) == 2:

#         for word in review_good_words:

#             if str(str_review[0][1]).find(word) != -1:

#                 return 1

#     return 0

            

# df['good_reviews'] = df['Reviews'].apply(fill_good_reviews)
# # empty set of unique cuisine

# unique_cuisines = set()



# # filling set of unique cuisine

# for row in data['Cuisine Style']:

#     if str(row) != 'nan':

#         cuisine_list = ast.literal_eval(row)

#         for cuisine in cuisine_list:

#             unique_cuisines.add(cuisine)



# # how many unique cuisines we have in dataset

# print(len(unique_cuisines))





# # counting amount of restraunts with each cuisine type

# cuisines_amount = {}



# # creating keys

# for cuisine in unique_cuisines:

#     cuisines_amount[cuisine] = 0



# # filling values

# for row in data['Cuisine Style']:

#     if str(row) != 'nan':

#         cuisine_list = ast.literal_eval(row)

#         for cuisine in cuisine_list:

#             cuisines_amount[cuisine] += 1



# #cuisines_amount



# # finding the most popular cuisine 

# max_count = 0

# cuisine = ''



# for key, value in cuisines_amount.items():

#     if value > max_count:

#         cuisine = key

#         max_count = value



# print(cuisine, max_count)



# # finding the most popular cuisine another solution



# cuisine = max(cuisines_amount, key = cuisines_amount.get)



# print(cuisine,cuisines_amount[cuisine])



# # finding rare cuisines



# rare_cuisines = []



# for key, value in cuisines_amount.items():

#     if value == 1:

#         rare_cuisines.append(key)



# print(rare_cuisines)
# creating columns reviews_na, difference bentween publication of to reviews and passed_time from latest review

# creating new dataframe using only one reading cycle

# DEGRADE







# #new dataframe collecting info

# df_reviews = pd.DataFrame(columns = ['reviews_na', 'difference', 'passed_time'])



# i = 0 # index counter



# for index, row in data.iterrows():

#     # initializing values

#     reviews_na = 0

#     difference = np.nan

#     passed_time = np.nan





#     str_review = ast.literal_eval(str(row['Reviews']).replace('nan','0'))

#     if str_review == [[], []]:

#     # this means that review is empty

#         reviews_na = 1



#     elif len(str_review[1]) == 1:

#     # there is only one review, we will calculate passed_time

#         passed_time = (datetime.today() - datetime.strptime(str_review[1][0],'%m/%d/%Y')).days



#     else:

#     # there are two reviews, we will calculate passed_time and difference

#         first_review_time = datetime.strptime(str_review[1][0],'%m/%d/%Y')

#         second_review_time = datetime.strptime(str_review[1][1],'%m/%d/%Y')

#         if first_review_time < second_review_time:

#             difference = (second_review_time - first_review_time).days

#             passed_time = (datetime.today() - second_review_time).days



#         else:

#             difference = (first_review_time - second_review_time).days

#             passed_time = (datetime.today() - first_review_time).days



#     df_reviews.loc[i] = [reviews_na, difference, passed_time]

#     i += 1





# data = pd.concat([data, df_reviews], axis=1)

# # searching the newest review date in dataset

# newest_review = datetime.strptime('01/01/1900','%m/%d/%Y')

# for item in last_review:

#     if item > newest_review:

#         newest_review = item

# print(newest_review.strftime('%Y-%m-%d'))
# # searching the eldest review date in dataset

# eldest_review = newest_review

# for item in old_review:

#     if item < eldest_review and item != datetime.strptime('01/01/1900','%m/%d/%Y'):

#         eldest_review = item

# print(eldest_review.strftime('%Y-%m-%d'))