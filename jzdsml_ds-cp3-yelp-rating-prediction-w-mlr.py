# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import pandas as pd



businesses = pd.read_json('../input/yelp_business.json', lines=True)

reviews = pd.read_json('../input/yelp_review.json', lines=True)

users = pd.read_json('../input/yelp_user.json', lines=True)

checkins = pd.read_json('../input/yelp_checkin.json', lines=True)

tips = pd.read_json('../input/yelp_tip.json', lines=True)

photos = pd.read_json('../input/yelp_photo.json', lines=True)
max_columns = 60

max_colwidth = 500
businesses.head()
reviews.head()
users.head()
checkins.head()
tips.head()
photos.head()
print(businesses.business_id.nunique())



print(list(reviews.columns))
print(users.describe())

#Another option

users.describe()
businesses[businesses['business_id'] == '5EvUIR4IzCWUOm0PsUZXjA']['stars']
# What feature, or column, do the DataFrames have in common?
df = pd.merge(businesses, reviews, how='left', on='business_id')

print(len(df))
df = pd.merge(df, users, how='left', on='business_id')

df = pd.merge(df, checkins, how='left', on='business_id')

df = pd.merge(df, tips, how='left', on='business_id')

df = pd.merge(df, photos, how='left', on='business_id')



print(df.columns)
features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']



df.drop(labels=features_to_remove, axis=1, inplace=True)
df.isna().any()
df.fillna({'weekday_checkins':0,

           'weekend_checkins':0,

           'average_tip_length':0,

           'number_tips':0,

           'average_caption_length':0,

           'number_pics':0},

          inplace=True)



df.isna().any()
df.corr()   
from matplotlib import pyplot as plt



# plot average_review_sentiment against stars here

plt.scatter(df.average_review_sentiment, df.stars)

plt.xlabel('Average Review Sentiment')

plt.ylabel('Ratings')

plt.title('Correlation between review sentiment and Yelp rating')

plt.show()
# plot average_review_length against stars here

plt.scatter(df.average_review_length, df.stars)

plt.xlabel('Average Review Length')

plt.ylabel('Ratings')

plt.title('Correlation between review length and Yelp rating')

plt.show()
# plot average_review_age against stars here

plt.scatter(df.average_review_age, df.stars)

plt.xlabel('Average Review Age')

plt.ylabel('Ratings')

plt.title('Correlation between review age and Yelp rating')

plt.show()

# plot number_funny_votes against stars here

plt.scatter(df.number_funny_votes, df.stars)

plt.xlabel('Number of Funny Votes')

plt.ylabel('Ratings')

plt.title('Correlation between funny votes and Yelp rating')

plt.show()

#Why do you think `average_review_sentiment` correlates so well with Yelp rating?
features = df[['average_review_length', 'average_review_age']]

ratings = df[['stars']]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)
print('Only 8.08% of the variance in Yelp rating is explained by chosen features.')
sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)
y_predicted = model.predict(X_test)

plt.scatter(y_test, y_predicted)

plt.xlabel('Yelp Rating')

plt.ylabel('Predicted Yelp Rating')

plt.show()
# subset of only average review sentiment

sentiment = ['average_review_sentiment']
# subset of all features that have a response range [0,1]

binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']
# subset of all features that vary on a greater range than [0,1]

numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']
# all features

all_features = binary_features + numeric_features



features = df[all_features]



X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)

print('All features explain 67.82% of the variance in Yelp rating.')
# add your own feature subset here



# I will choose from the features indicating larger correlation (> 0.1) with stars in the df.corr() table

feature_subset =['average_review_age', 'average_review_length', 'average_review_sentiment']

features = df[feature_subset]



X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)
print('My chosen features explain 64.96% of the variance in Yelp rating. A huge improvement from previous results!')
import numpy as np



# take a list of features to model as a parameter

def model_these_features(feature_list):

    

    # #select the dataframe with stars and features 

    ratings = df.loc[:,'stars']

    features = df.loc[:,feature_list]

    

    # #split train and test data

    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

    

    # don't worry too much about these lines, just know that they allow the model to work when

    # we model on just one feature instead of multiple features. Trust us on this one :)

    if len(X_train.shape) < 2:

        X_train = np.array(X_train).reshape(-1,1)

        X_test = np.array(X_test).reshape(-1,1)

    

    # #train

    model = LinearRegression()

    model.fit(X_train,y_train)

    

    # #get score 

    print('Train Score:', model.score(X_train,y_train))

    print('Test Score:', model.score(X_test,y_test))

    

    # print the model features and their corresponding coefficients, from most predictive to least predictive

    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))

    

    # #get prediced data

    y_predicted = model.predict(X_test)

    

    # #compare predicted data with actual data

    plt.scatter(y_test,y_predicted)

    plt.xlabel('Yelp Rating')

    plt.ylabel('Predicted Yelp Rating')

    plt.ylim(1,5)

    plt.show()
# create a model on sentiment here

model_these_features(sentiment)
# create a model on all binary features here

model_these_features(binary_features)
# create a model on all numeric features here

model_these_features(numeric_features)
# create a model on all features here

model_these_features(all_features)
# create a model on your feature subset here

model_these_features(feature_subset)

print(all_features)
features = df.loc[:,all_features]

ratings = df.loc[:,'stars']

X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)

model = LinearRegression()

model.fit(X_train,y_train)
pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])
danielles_delicious_delicacies = np.array([0,1,1,1,1,1,10,2,3,10,10,1200,0.9,3,6,5,50,3,50,1800,12,123,0.5,0,0]).reshape(1,-1)
model.predict(danielles_delicious_delicacies)