import numpy as np 

import pandas as pd 

import json

import os

from pandas import json_normalize #package for flattening json in pandas df

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_colwidth', 1)

pd.options.display.max_rows = 999
# Load the review data

with open("../input/yelp-dataset/yelp_academic_dataset_review.json") as reviews:

    

    words = ["poison","diar","sick","puk","salmonella","chunder","spew","vom","throw up","threw up","nause"]

    count = 0 

    

    highRiskReviews = pd.DataFrame(columns=['business_id', 'date', 'text'])

    normalReviews = pd.DataFrame(columns=['business_id', 'date', 'text'])

    

    # Iterate through the reviews and save high risk reviews to highRiskReviews

    for index,review in enumerate(reviews):

        reviewData = json.loads(review)

        

        # if text contains one of the key words, and is 1 star or less then append to the data frame

        if any(word in reviewData['text'] for word in words) and reviewData['stars'] < 2:

            highRiskReviews = highRiskReviews.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']}, ignore_index=True)

        

        # because of the size of the data set, pick approx one in every 200, without key words as normal reviews

        if np.random.randint(1,180) == 1 and not any(word in reviewData['text'] for word in words):

            normalReviews = normalReviews.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']}, ignore_index=True)

            count += 1

        

    print(highRiskReviews.info())

    print(normalReviews.info())
highRiskReviews.sample(n=5)
normalReviews.sample(n=5)
def formatDataFromID(business_ids, highRisk = 0):

    

    X = pd.DataFrame()



    with open("../input/yelp-dataset/yelp_academic_dataset_business.json") as businesses:

        

        for business in businesses:



            businessData = json.loads(business)



            if businessData['business_id'] in business_ids:

                if businessData['categories']:

                    if 'Food' in businessData['categories'] or 'Restaurants' in businessData['categories']:



                        # Drop columns that are unlikely to be useful features (think about dropping review count and number of stars as this will leak data)

                        columnsToDrop = ["hours", "name", "address", "city", "state", "postal_code", "latitude", "longitude", "is_open", "BusinessParking", "GoodForMeal"]

                        [businessData.pop(column, None) for column in columnsToDrop]



                        # Add in category data, represented as a string list

                        categoryColumns = businessData["categories"].split(",")

                        for category in categoryColumns:

                            businessData['category_' + category.strip()] = 1



                        # Add in attribute data, represented as a string dict

                        if businessData["attributes"]:

                            for key,value in businessData["attributes"].items():

                                if key == "Ambience":

                                    if businessData["attributes"]["Ambience"] != "None":

                                        ambienceType = businessData["attributes"].get(key)

                                        for key,value in eval(ambienceType).items():

                                            if value == 0:

                                                businessData['ambience_' + key] = 0

                                            else:

                                                businessData['ambience_' + key] = 1



                        # Assign the target variable (whether the restaurant has a high risk of food poisoning)

                        businessData['highRisk'] = highRisk



                        X = X.append(json_normalize(businessData, sep='_'), ignore_index=True)

        return X
highRiskBusinesses = highRiskReviews['business_id'].tolist()

normalBusinesses = normalReviews['business_id'].tolist()



X_highRisk = formatDataFromID(highRiskBusinesses, 1)

X_normal = formatDataFromID(normalBusinesses)
print("There are {} high risk businesses with an average rating of {:.1f} stars.".format(highRiskReviews['business_id'].nunique(), X_highRisk['stars'].mean()))

print("There are {} normal businesses with an average rating of {:.1f} stars.".format(normalReviews['business_id'].nunique(), X_normal['stars'].mean()))
# Merge the high and normal data together

X = pd.concat([X_highRisk, X_normal], ignore_index=True)



# Ambience and categories columns have already been parsed, so we can drop them

X_clean = X.drop(['attributes_Ambience', 'categories', 'category_Restaurants', 'business_id','attributes_BusinessParking','attributes_BikeParking', 'attributes_Music'], axis=1)



# Drop columns with more than 90% NaN values, these are likely to be spelling mistakes or obscure features

cols_with_missing = [col for col in X.columns if X[col].isnull().sum() > X.shape[0]*0.9]

X_clean = X_clean.drop(cols_with_missing, axis=1)



# Replace NaNs with 0's as they are almost all categorical data

X_clean = X_clean.fillna(0)



# Find object columns

objects = (X_clean.dtypes == 'object')

object_cols = list(objects[objects].index)



# Drop high cardinality objects

low_cardinality_cols = [col for col in object_cols if X_clean[col].nunique() < 10]



# Columns to be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



# Change the column data to be strings, this doesn't matter as they will be OH encoded

for col in object_cols:

    X_clean[col] = X_clean[col].astype(str)



X_clean = X_clean.drop(high_cardinality_cols, axis=1)



# OH encode the data

OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(X_clean[low_cardinality_cols]))



# Put the index back in

OH_cols.index = X_clean.index



# Drop the columns that have been OH encoded

num_X = X_clean.drop(low_cardinality_cols, axis=1)

OH_X = pd.concat([num_X, OH_cols], axis=1)
OH_X.info()
y = OH_X['highRisk']

OH_X= OH_X.drop('highRisk', axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(OH_X, y, train_size=0.85, test_size=0.15, random_state=1)
# Define the model

model = XGBRegressor(n_estimators=500)



# early stopping rounds will stop when the prediction stops improving. This is compares values by evaluating X_valid, y_valid

model.fit(X_train, y_train,

        early_stopping_rounds=5,

        eval_set=[(X_valid, y_valid)],

        verbose=False)



predictions = model.predict(X_valid)

print("Mean absolute error: {:.2f}".format(mean_absolute_error(predictions, y_valid)))
# Find the restaurants with the highest and lowest predicted risk of food poisoning

highestRiskRestaurant = predictions.max()

lowestRiskRestaurant = predictions.min()



print("The highest risk restaurant has a food poisoning prediction of {:.0f}%.".format(highestRiskRestaurant*100))

print("The lowest risk restaurant has a food poisoning prediction of {:.0f}%.".format(lowestRiskRestaurant*100))



# Locate these in predictions array

highRiskIndexInPredictionsArray = np.where(predictions == highestRiskRestaurant)[0]

lowRiskIndexInPredictionsArray = np.where(predictions == lowestRiskRestaurant)[0]



# Find the row index in the validation data, to find the restaurant_id

highRiskRow = X_valid.iloc[highRiskIndexInPredictionsArray]

lowRiskRow = X_valid.iloc[lowRiskIndexInPredictionsArray]
highRiskReviewBusinessID = X.iloc[highRiskRow.index]['business_id']

lowRiskReviewBusinessID = X.iloc[lowRiskRow.index]['business_id']
with open("../input/yelp-dataset/yelp_academic_dataset_review.json") as reviews:

    

    highestRiskReviews = pd.DataFrame(columns=['text'])

    lowestRiskReviews = pd.DataFrame(columns=['text'])

    

    # Iterate through the reviews and save high risk reviews to highRiskReviews

    for review in reviews:

        reviewData = json.loads(review)

        if reviewData['business_id'] == highRiskReviewBusinessID.values[0]:

            highestRiskReviews = highestRiskReviews.append({'text': reviewData['text']}, ignore_index=True)

        

        if reviewData['business_id'] == lowRiskReviewBusinessID.values[0]:

            lowestRiskReviews = lowestRiskReviews.append({'text': reviewData['text']}, ignore_index=True)
X.loc[X['business_id'] == lowRiskReviewBusinessID.values[0]]
lowestRiskReviews.head()
X.loc[X['business_id'] == highRiskReviewBusinessID.values[0]]
highestRiskReviews.head()