# Importing modules.
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns 
import os

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

%matplotlib inline
#Setting the conditions for experiments.
random_seed = 42
current_date = pd.to_datetime('29/08/2020')
!pip freeze > requirements.txt
data_directory = '/kaggle/input/sf-dst-restaurant-rating/'
# Importing datasets.
data_train = pd.read_csv(data_directory+'/main_task.csv')
data_test = pd.read_csv(data_directory+'/kaggle_task.csv')
sample_submission = pd.read_csv(data_directory+'/sample_submission.csv')
# Checkin the dataset for training.
data_train.info()
data_train.sample(5)
# Checkin the dataset for prediction.
data_test.info()
data_test.sample(5)
# Checkin the dataset for submission.
sample_submission.info()
sample_submission.sample(5)
# Merging the datasets.
data_train['Sample'] = 1
data_test['Sample'] = 0
data_test['Rating'] = 0
data = data_test.append(data_train, sort=False).reset_index(drop=True)
# Checking the merged data.
data.info()
data.sample(5)
# Checking the data.
data['Cuisine Style'][1]
# Checking the data.
data['Reviews'][1]
# Counting unique values.
data.nunique(dropna = False)
# Checking for missing values.
data.isna().sum()
# Counting values.
data['Restaurant_id'].value_counts()
# Checking the frequency distribution.
plt.rcParams['figure.figsize'] = (13,8)
data['Restaurant_id'].value_counts().hist(bins=100)
# Dropping the Restaurant_id column.
data.drop(['Restaurant_id'], inplace = True, axis = 1)
# Counting values.
data['City'].value_counts()
#Counting unique cities.
data['City'].nunique()
# Checking the frequency distribution.
data['City'].value_counts(ascending=True).plot(kind='barh')
# Checking the Rating distribution by city.
fig, ax = plt.subplots(figsize = (15, 5))

sns.boxplot(x='City', y='Rating',data=data.loc[
    data.loc[:, 'City'].isin(data.loc[:, 'City'].value_counts().index[:])
],ax=ax)

plt.xticks(rotation=45)
ax.set_title('Boxplot for City')

plt.show()
# Checking the Ranking distribution by city.
for x in (data['City'].value_counts())[0:10].index:
    data['Ranking'][data['City'] == x].hist(bins=100)
plt.show()
# Creating dictionary with number of restaurants in the city.
res_count = {
    'Paris': 17593,
    'Stockholm': 3131,
    'London': 22366,
    'Berlin': 8110, 
    'Munich': 3367,
    'Oporto': 2060, 
    'Milan': 7940,
    'Bratislava': 1331,
    'Vienna': 4387, 
    'Rome': 12086,
    'Barcelona': 10086,
    'Madrid': 11562,
    'Dublin': 2706,
    'Brussels': 3703,
    'Zurich': 1901,
    'Warsaw': 3210,
    'Budapest': 3445, 
    'Copenhagen': 2637,
    'Amsterdam': 4189,
    'Lyon': 2833,
    'Hamburg': 3501, 
    'Lisbon': 4985,
    'Prague': 5850,
    'Oslo': 1441, 
    'Helsinki': 1661,
    'Edinburgh': 2248,
    'Geneva': 1753,
    'Ljubljana': 647,
    'Athens': 2814,
    'Luxembourg': 759,
    'Krakow': 1832       
}
# Mapping the dataset with dictionary.
data['Restaurants Count'] = data['City'].map(res_count)
data['Restaurants Count']
# Converting feature to dummy variables.
data = pd.get_dummies(data, columns=['City',], dummy_na=True)
data.sample(5)
# Counting values.
data['Cuisine Style'].value_counts()
# Creating a binary variable for missing values.
data['Cuisine Style NAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
# Filling the missing values.
data['Cuisine Style'] = data['Cuisine Style'].fillna('no cuisine provided')
# Lowering the cases.
data['Cuisine Style'] = data['Cuisine Style'].str.lower() 
data['Cuisine Style']
# Converting string values to lists.
def cuisines_to_list(string):
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace("'", '')
    return string.split(', ')

data['Cuisine Style'] = data['Cuisine Style'].apply(cuisines_to_list)
# Checking the value type.
type(data['Cuisine Style'][0])
# Creating a new feature.
data['Number of Cuisines'] = data['Cuisine Style'].apply(lambda x: len(x))
data['Number of Cuisines'].value_counts()
# Creating a set of unique cuisine styles.
cuisines_set = set()

for restaraunt_cuisines in data['Cuisine Style']:
    for cuisine in restaraunt_cuisines:
        cuisines_set.add(cuisine)
        
len(cuisines_set)
# Counting cuisine styles.
cuisines_counter = dict.fromkeys(cuisines_set, 0)

for cuisine in cuisines_set:
    for restaraunt_cuisines in data['Cuisine Style']:
        if cuisine in restaraunt_cuisines:
            cuisines_counter[cuisine] += 1
            
cuisines_counter = pd.Series(cuisines_counter)
cuisines_counter.sort_values(ascending=False)[0:20]
# Converting a feature to dummy variables.
data_cuisines = pd.get_dummies(data['Cuisine Style'].apply(pd.Series).stack()).sum(level=0)
data = pd.merge(data, data_cuisines, left_index=True, right_index=True)
data.sample(5)
# Checking the Ranking distribution.
data['Ranking'].hist(bins=100)
# Creating a new feature.
data['Relative Ranking'] = data['Ranking'] / data['Restaurants Count']
data['Relative Ranking']
# Counting values.
data['Price Range'].value_counts(dropna=False)
# Creating a binary variable for missing values.
data['Price Range NAN'] = pd.isna(data['Price Range']).astype('uint8')
# Checking the URLs of restaraunts with no price range.
data.query('`Price Range NAN` == 1')['URL_TA'].tolist()[0:11]
# Filling the missing values.
data['Price Range'] = data['Price Range'].fillna('$$ - $$$')
# Replacing the object values with numeric values
price_transform_dict = {'$':1,'$$ - $$$':2,'$$$$':3}
data['Price Range'] = data['Price Range'].map(lambda x: price_transform_dict.get(x,x))

data['Price Range'].value_counts()
# Checking the Rating distribution by price range.
fig, ax = plt.subplots(figsize = (15, 5))

sns.boxplot(x='Price Range', y='Rating',data=data.loc[
    data.loc[:, 'Price Range'].isin(data.loc[:, 'Price Range'].value_counts().index[:])
],ax=ax)

plt.xticks(rotation=45)
ax.set_title('Boxplot for Price Range')

plt.show()
# Counting values.
data['Number of Reviews'].value_counts(dropna=False)
# Creating a binary variable for missing values.
data['Number of Reviews NAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# Checking the frequency distribution.
data['Number of Reviews'].hist(bins=100)
# Checking the URLs of restaraunts with no number of reviews.
data.query('`Number of Reviews NAN` == 1')['URL_TA'].tolist()[0:11]
# Filling the missing values.
data['Number of Reviews'] = data['Number of Reviews'].fillna(0)
# Creating a new feature.
data['Reviews to Ranking Ratio'] = data['Number of Reviews'] / data['Ranking']
data['Reviews to Ranking Ratio']
# Checking the data.
data['Reviews'].head(20)
# Filling the missing values.
data['Reviews'] = data['Reviews'].fillna('[[], []]')
# Extracting the lists with dates of reviews.
def reviews_to_dates_list(string):
    if string == '[[], []]':
        return []
    else:
        string = string.replace(']]', '')
        string = string.replace("'", '')
        string = string.split('], [')[1]
        string = string.split(', ')
        return string
    
data['Dates of Reviews'] = data['Reviews'].apply(reviews_to_dates_list)
# Checking the data.
data['Dates of Reviews'].head(20)
# Splitting the dates of reviews.
data[['Latest Review Date','Pre-Latest Review Date']] = pd.DataFrame(
    data['Dates of Reviews'].tolist(), index= data.index)
# Converting values to datetime.
data['Latest Review Date'] = pd.to_datetime(data['Latest Review Date'])
data['Pre-Latest Review Date'] = pd.to_datetime(data['Pre-Latest Review Date'])
# Creating a binary variables for missing values.
data['Latest Review Date NAN'] = pd.isna(data['Latest Review Date']).astype('uint8')
data['Pre-Latest Review Date NAN'] = pd.isna(data['Pre-Latest Review Date']).astype('uint8')
# Converting the features to float.
data['Latest Review Date'] = pd.to_timedelta(
    data['Latest Review Date']).dt.total_seconds()
data['Pre-Latest Review Date'] = pd.to_timedelta(
    data['Pre-Latest Review Date']).dt.total_seconds()
# Filling the missing values.
data['Latest Review Date'] = data['Latest Review Date'].fillna(0)
data['Pre-Latest Review Date'] = data['Pre-Latest Review Date'].fillna(0)
# Creating a new feature.
data['Timedelta for Reviews'] = data['Latest Review Date'] - data['Pre-Latest Review Date']
data['Timedelta for Reviews'].value_counts().sort_index()
# Creating a new feature.
current_date_float = pd.to_timedelta(pd.Series(current_date)).dt.total_seconds()[0]
data['Timedelta Today - Last Review'] = current_date_float - data['Latest Review Date']
data['Timedelta Today - Last Review']
# Cleaning data to get content of latests reviews.
def reviews_to_latest_review(string):
    if string == '[[], []]':
        return float ('NaN')
    else:
        string = string.replace(']]', '')
        string = string.replace("'", '')
        string = string.split('], [')[0]
        string = string.replace('[[', '')
        string = string.split(', ')[0]
        return string

data['Latest Review Content'] = data['Reviews'].apply(reviews_to_latest_review)
data['Latest Review Content']
# Cleaning data to get content of pre-latests reviews.
def reviews_to_pre_latest_review(string):
    if string == '[[], []]':
        return float ('NaN')
    else:
        string = string.replace(']]', '')
        string = string.replace("'", '')
        string = string.split('], [')[0]
        string = string.replace('[[', '')
        string = string.split(', ')[-1]
        return string
    
data['Pre-Latest Review Content'] = data['Reviews'].apply(reviews_to_pre_latest_review)
data['Pre-Latest Review Content']
# Creating a binary variables for missing values.
data['Latest Review Content NAN'] = pd.isna(data['Latest Review Content']).astype('uint8')
data['Pre-Latest Review Content NAN'] = pd.isna(data['Pre-Latest Review Content']).astype('uint8')
# Filling the missing values.
data['Latest Review Content'] = data['Latest Review Content'].fillna('no review provided')
data['Pre-Latest Review Content'] = data['Pre-Latest Review Content'].fillna('no review provided')
# Lowering the cases.
data['Latest Review Content'] = data['Latest Review Content'].str.lower() 
data['Pre-Latest Review Content'] = data['Pre-Latest Review Content'].str.lower()
# Checking the data.
pd.set_option('display.min_rows', 200)
pd.DataFrame(data['Latest Review Content'].str.split().tolist()).stack().value_counts().head(200)
# Checking the data.
pd.DataFrame(data['Pre-Latest Review Content'].str.split().tolist()).stack().value_counts().head(200)
# Establishing lists of words.
good_words_list = [
    'great', 'good', 'nice', 'best', 'excellent', 'delicious', 
    'lovely', 'friendly', 'tasty', 'amazing', 'fantastic', 
    'perfect', 'wonderful', 'gem', 'fresh', 'decent', 'cozy', 
    'pleasant', 'love', 'awesome', 'beautiful', 'yummy', 
    'fabulous', 'superb', 'fine', 'brilliant', 'cute', 'super' 
    'favourite', 'enjoyable', 'favorite', 'outstanding', 
    'pretty', 'affordable', 'charming', 'delightful', 
    'unique', 'incredible', 'solid', 'exceptional'
]

bad_words_list = [
    'bad', 'poor', 'worst', 'terrible', 'disappointing', 
    'overpriced', 'rude', 'avoid', 'awful', 'disappointed', 
    'horrible', 'mediocre', 'cold'
]
# Semantic analysis.
latest_good_set = set()
latest_bad_set = set()
pre_latest_good_set = set()
pre_latest_bad_set = set()

for review in data['Latest Review Content']:
    for word in good_words_list:
        if word in review:
            latest_good_set.add(data[data['Latest Review Content'] == review].index[0])
    for word in bad_words_list:
        if word in review:
            latest_bad_set.add(data[data['Latest Review Content'] == review].index[0])  

for review in data['Pre-Latest Review Content']:
    for word in good_words_list:
        if word in review:
            pre_latest_good_set.add(data[data['Pre-Latest Review Content'] == review].index[0]) 
    for word in bad_words_list:
        if word in review:
            pre_latest_bad_set.add(data[data['Pre-Latest Review Content'] == review].index[0])            
# Creating columns for semantic binary variables.
data['Latest Review Good'] = 0
data['Pre-Latest Review Good'] = 0
data['Latest Review Bad'] = 0
data['Pre-Latest Review Bad'] = 0
# Filling semantic binary variables.
for value in latest_good_set:
    data ['Latest Review Good'][value] = 1

for value in pre_latest_good_set:
    data ['Pre-Latest Review Good'][value] = 1    

for value in latest_bad_set:
    data ['Latest Review Bad'][value] = 1

for value in pre_latest_bad_set:
    data ['Pre-Latest Review Bad'][value] = 1
# Checking the data.
data['URL_TA'][1]
# Dropping the URL_TA column.
data.drop(['URL_TA'], inplace = True, axis = 1)
# Counting values.
pd.set_option('display.min_rows', 10)
data['ID_TA'].value_counts()
# Checking the data.
data[data['ID_TA'] == 'd4600226']
# Counting duplicates.
data['ID_TA'].duplicated().value_counts()
# Checking the frequency distribution.
plt.rcParams['figure.figsize'] = (13,8)
data['Rating'].value_counts(ascending=True).sort_index().plot(kind='barh')
# Checking the Ranking distribution by rating.
fig, ax = plt.subplots(figsize = (15, 5))
sns.boxplot(x = 'Rating', 
            y = 'Ranking', 
            data = data.loc[data.loc[:, 'Rating'].isin(
                data.loc[:, 'Rating'].value_counts().index[:]
            )],ax=ax)
ax.set_title('Boxplot for Rating')
plt.show()
# Using the correlation matrix (in absolute value).
data.corr().abs().sort_values(by='Rating', ascending=False)
# Checking the data.
data.info()
# Dropping object columns.
object_columns = [s for s in data.columns if data[s].dtypes == 'object']
data.drop(object_columns, axis = 1, inplace=True)
# Checking the data.
data.info()
# Normalizing data (except target variable).
# data_no_rating = data.loc[:, data.columns != 'Rating']

# data_no_rating[data_no_rating.columns] = pd.DataFrame(
#     min_max_scaler.fit_transform(data_no_rating[data_no_rating.columns]))

# data_no_rating['Rating'] = data['Rating']

# data = data_no_rating
data_train = pd.read_csv(data_directory+'/main_task.csv')
data_test = pd.read_csv(data_directory+'/kaggle_task.csv')

data_train['Sample'] = 1
data_test['Sample'] = 0
data_test['Rating'] = 0

data = data_test.append(data_train, sort=False).reset_index(drop=True)

data.info()
data.sample(5)
def preproc_data(data_input):
    
    data_output = data_input.copy()
    
    ###########################################
    ################## CITY ###################
    ###########################################

    # Creating dictionary with number of restaurants in the city.
    res_count = {
        'Paris': 17593,
        'Stockholm': 3131,
        'London': 22366,
        'Berlin': 8110, 
        'Munich': 3367,
        'Oporto': 2060, 
        'Milan': 7940,
        'Bratislava': 1331,
        'Vienna': 4387, 
        'Rome': 12086,
        'Barcelona': 10086,
        'Madrid': 11562,
        'Dublin': 2706,
        'Brussels': 3703,
        'Zurich': 1901,
        'Warsaw': 3210,
        'Budapest': 3445, 
        'Copenhagen': 2637,
        'Amsterdam': 4189,
        'Lyon': 2833,
        'Hamburg': 3501, 
        'Lisbon': 4985,
        'Prague': 5850,
        'Oslo': 1441, 
        'Helsinki': 1661,
        'Edinburgh': 2248,
        'Geneva': 1753,
        'Ljubljana': 647,
        'Athens': 2814,
        'Luxembourg': 759,
        'Krakow': 1832       
    }

    # Mapping the dataset with dictionary.
    data_output['Restaurants Count'] = data_output['City'].map(res_count)

    # Converting feature to dummy variables.
    data_output = pd.get_dummies(data_output, columns=['City',], dummy_na=True)

    ###########################################
    ################ CUISINE ##################
    ###########################################

    # Creating a binary variable for missing values.
    data_output['Cuisine Style NAN'] = pd.isna(data_output['Cuisine Style']).astype('uint8')

    # Filling the missing values.
    data_output['Cuisine Style'] = data_output['Cuisine Style'].fillna('no cuisine provided')

    # Lowering the cases.
    data_output['Cuisine Style'] = data_output['Cuisine Style'].str.lower() 

    # Converting string values to lists.
    def cuisines_to_list(string):
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.replace("'", '')
        return string.split(', ')

    data_output['Cuisine Style'] = data_output['Cuisine Style'].apply(cuisines_to_list)

    # Creating a new feature.
    data_output['Number of Cuisines'] = data_output['Cuisine Style'].apply(lambda x: len(x))

    # Creating a set of unique cuisine styles.
    cuisines_set = set()

    for restaraunt_cuisines in data_output['Cuisine Style']:
        for cuisine in restaraunt_cuisines:
            cuisines_set.add(cuisine)

    # Counting cuisine styles.
    cuisines_counter = dict.fromkeys(cuisines_set, 0)

    for cuisine in cuisines_set:
        for restaraunt_cuisines in data_output['Cuisine Style']:
            if cuisine in restaraunt_cuisines:
                cuisines_counter[cuisine] += 1

    cuisines_counter = pd.Series(cuisines_counter)

    # Converting a feature to dummy variables.
    data_output_cuisines = pd.get_dummies(data_output['Cuisine Style'].apply(pd.Series).stack()).sum(level=0)
    data_output = pd.merge(data_output, data_output_cuisines, left_index=True, right_index=True)

    ############################################
    ################# RANKING ##################
    ############################################

    # Creating a new feature.
    data_output['Relative Ranking'] = data_output['Ranking'] / data_output['Restaurants Count']

    ##################################################
    ################## PRICE RANGE ###################
    ##################################################

    # Creating a binary variable for missing values.
    data_output['Price Range NAN'] = pd.isna(data_output['Price Range']).astype('uint8')

    # Filling the missing values.
    data_output['Price Range'] = data_output['Price Range'].fillna('$$ - $$$')

    # Replacing the object values with numeric values
    price_transform_dict = {'$':1,'$$ - $$$':2,'$$$$':3}
    data_output['Price Range'] = data_output['Price Range'].map(lambda x: price_transform_dict.get(x,x))

    ###################################################
    ############## NUMBER OF REVIEWS ##################
    ###################################################

    # Creating a binary variable for missing values.
    data_output['Number of Reviews NAN'] = pd.isna(data_output['Number of Reviews']).astype('uint8')

    # Filling the missing values.
    data_output['Number of Reviews'] = data_output['Number of Reviews'].fillna(0)

    # Creating a new feature.
    data_output['Reviews to Ranking Ratio'] = data_output['Number of Reviews'] / data_output['Ranking']

    #########################################
    ############## REVIEWS ##################
    #########################################

    # Filling the missing values.
    data_output['Reviews'] = data_output['Reviews'].fillna('[[], []]')

    # Extracting the lists with dates of reviews.
    def reviews_to_dates_list(string):
        if string == '[[], []]':
            return []
        else:
            string = string.replace(']]', '')
            string = string.replace("'", '')
            string = string.split('], [')[1]
            string = string.split(', ')
            return string

    data_output['Dates of Reviews'] = data_output['Reviews'].apply(reviews_to_dates_list)

    # Splitting the dates of reviews.
    data_output[['Latest Review Date','Pre-Latest Review Date']] = pd.DataFrame(
        data_output['Dates of Reviews'].tolist(), index= data_output.index)

    # Converting values to datetime.
    data_output['Latest Review Date'] = pd.to_datetime(data_output['Latest Review Date'])
    data_output['Pre-Latest Review Date'] = pd.to_datetime(data_output['Pre-Latest Review Date'])

    # Creating a binary variables for missing values.
    data_output['Latest Review Date NAN'] = pd.isna(data_output['Latest Review Date']).astype('uint8')
    data_output['Pre-Latest Review Date NAN'] = pd.isna(data_output['Pre-Latest Review Date']).astype('uint8')

    # Converting the features to float.
    data_output['Latest Review Date'] = pd.to_timedelta(
        data_output['Latest Review Date']).dt.total_seconds()
    data_output['Pre-Latest Review Date'] = pd.to_timedelta(
        data_output['Pre-Latest Review Date']).dt.total_seconds()

    # Filling the missing values.
    data_output['Latest Review Date'] = data_output['Latest Review Date'].fillna(0)
    data_output['Pre-Latest Review Date'] = data_output['Pre-Latest Review Date'].fillna(0)

    # Creating a new feature.
    data_output['Timedelta for Reviews'] = data_output['Latest Review Date'] - data_output['Pre-Latest Review Date']

    # Creating a new feature.
    current_date_float = pd.to_timedelta(pd.Series(current_date)).dt.total_seconds()[0]
    data_output['Timedelta Today - Last Review'] = current_date_float - data_output['Latest Review Date']

    # Cleaning data to get content of latests reviews.
    def reviews_to_latest_review(string):
        if string == '[[], []]':
            return float ('NaN')
        else:
            string = string.replace(']]', '')
            string = string.replace("'", '')
            string = string.split('], [')[0]
            string = string.replace('[[', '')
            string = string.split(', ')[0]
            return string

    data_output['Latest Review Content'] = data_output['Reviews'].apply(reviews_to_latest_review)

    # Cleaning data to get content of pre-latests reviews.
    def reviews_to_pre_latest_review(string):
        if string == '[[], []]':
            return float ('NaN')
        else:
            string = string.replace(']]', '')
            string = string.replace("'", '')
            string = string.split('], [')[0]
            string = string.replace('[[', '')
            string = string.split(', ')[-1]
            return string

    data_output['Pre-Latest Review Content'] = data_output['Reviews'].apply(reviews_to_pre_latest_review)

    # Creating a binary variables for missing values.
    data_output['Latest Review Content NAN'] = pd.isna(data_output['Latest Review Content']).astype('uint8')
    data_output['Pre-Latest Review Content NAN'] = pd.isna(data_output['Pre-Latest Review Content']).astype('uint8')

    # Filling the missing values.
    data_output['Latest Review Content'] = data_output['Latest Review Content'].fillna('no review provided')
    data_output['Pre-Latest Review Content'] = data_output['Pre-Latest Review Content'].fillna('no review provided')

    # Lowering the cases.
    data_output['Latest Review Content'] = data_output['Latest Review Content'].str.lower() 
    data_output['Pre-Latest Review Content'] = data_output['Pre-Latest Review Content'].str.lower()

    # Establishing lists of words.
    good_words_list = [
        'great', 'good', 'nice', 'best', 'excellent', 'delicious', 
        'lovely', 'friendly', 'tasty', 'amazing', 'fantastic', 
        'perfect', 'wonderful', 'gem', 'fresh', 'decent', 'cozy', 
        'pleasant', 'love', 'awesome', 'beautiful', 'yummy', 
        'fabulous', 'superb', 'fine', 'brilliant', 'cute', 'super' 
        'favourite', 'enjoyable', 'favorite', 'outstanding', 
        'pretty', 'affordable', 'charming', 'delightful', 
        'unique', 'incredible', 'solid', 'exceptional'
    ]

    bad_words_list = [
        'bad', 'poor', 'worst', 'terrible', 'disappointing', 
        'overpriced', 'rude', 'avoid', 'awful', 'disappointed', 
        'horrible', 'mediocre', 'cold'
    ]

    # Semantic analysis.
    latest_good_set = set()
    latest_bad_set = set()
    pre_latest_good_set = set()
    pre_latest_bad_set = set()

    for review in data_output['Latest Review Content']:
        for word in good_words_list:
            if word in review:
                latest_good_set.add(data_output[data_output['Latest Review Content'] == review].index[0])
        for word in bad_words_list:
            if word in review:
                latest_bad_set.add(data_output[data_output['Latest Review Content'] == review].index[0])  

    for review in data_output['Pre-Latest Review Content']:
        for word in good_words_list:
            if word in review:
                pre_latest_good_set.add(data_output[data_output['Pre-Latest Review Content'] == review].index[0]) 
        for word in bad_words_list:
            if word in review:
                pre_latest_bad_set.add(data_output[data_output['Pre-Latest Review Content'] == review].index[0])            

    # Creating columns for semantic binary variables.
    data_output['Latest Review Good'] = 0
    data_output['Pre-Latest Review Good'] = 0
    data_output['Latest Review Bad'] = 0
    data_output['Pre-Latest Review Bad'] = 0

    # Filling semantic binary variables.
    for value in latest_good_set:
        data_output ['Latest Review Good'][value] = 1

    for value in pre_latest_good_set:
        data_output ['Pre-Latest Review Good'][value] = 1    

    for value in latest_bad_set:
        data_output ['Latest Review Bad'][value] = 1

    for value in pre_latest_bad_set:
        data_output ['Pre-Latest Review Bad'][value] = 1

    ############################################
    ############## FINALIZING ##################
    ############################################

    # Dropping object columns.
    object_columns = [s for s in data_output.columns if data_output[s].dtypes == 'object']
    data_output.drop(object_columns, axis = 1, inplace=True)

    # Normalizing data.
    #data_no_rating = data_output.loc[:, data_output.columns != 'Rating']

    #data_no_rating[data_no_rating.columns] = pd.DataFrame(
    #    min_max_scaler.fit_transform(data_no_rating[data_no_rating.columns]))

    #data_no_rating['Rating'] = data_output['Rating']

    #data_output = data_no_rating
    
    return data_output
data_preproc = preproc_data(data)
data_preproc.info()
data_preproc.sample(10)
# Dividing the dataset for testing
train_data = data_preproc.query('Sample == 1').drop(['Sample'], axis=1)
test_data = data_preproc.query('Sample == 0').drop(['Sample'], axis=1)

# Target dataset.
y = train_data.Rating.values
X = train_data.drop(['Rating'], axis=1)

#Allocating 20 percent of the dataset for validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_seed)
# Creation of a model.
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state = random_seed)

# Training a model on a test dataset.
model.fit(X_train, y_train)

# Defining a function for rating correction.
def rating_correction (x):
    return np.round(x * 2) / 2

# Predicting the rating in a test sample.
y_pred = np.array([rating_correction(x) for x in model.predict(X_test)])
# Mean Absolute Error (MAE) estimation.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Evaluating the importance of variables for the forecast.
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
# Checking the data.
test_data.info()
test_data.sample(10)
# Dropping Rating column.
test_data = test_data.drop(['Rating'], axis=1)
# Checking the data.
sample_submission
# Predicting the rating in a test sample.
predict_submission = np.array([rating_correction(x) for x in model.predict(test_data)])
# Checking the data.
predict_submission
# Result submission.
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.sample(10)