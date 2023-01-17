import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
Hotel = pd.read_csv("../input/7282_1.csv")
print ("There are " + str(Hotel.shape[0]) + " rows and "+ str(Hotel.shape[1]) + " columns in this data set.")
Hotel.head(5)
Hotel.isnull().sum()
#change the names of a few variables to avoid confusion
Hotel=Hotel.rename(columns={'province': 'state', 'postalCode': 'zipcode', 'reviews.userProvince':'reviews.userState', }) 
#drop the variables that are not useful to us
Hotel=Hotel.drop(["reviews.doRecommend","reviews.id", "country"], axis=1)
Hotel=Hotel.drop(city=="Curitiba",)
statesList = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
#import the complete list of 50 states abbreviations and the uszipcode package
from uszipcode import ZipcodeSearchEngine
search = ZipcodeSearchEngine()
# validate the state names by cross-referrencing the zipcode
for i in range(Hotel.shape[0]):
    if Hotel.loc[i,'state']not in statesList:
        Hotelzip=search.by_zipcode(Hotel.loc[i, 'zipcode'])
        Hotel.loc[i,'state']=Hotelzip.State

for i in range(Hotel.shape[0]):
    if Hotel.loc[i,'reviews.userState']not in statesList:
        userzip=search.by_zipcode(Hotel.loc[i, 'zipcode'])
        Hotel.loc[i,'reviews.userState']=userzip.State
print("There are "+str(Hotel['name'].nunique())+ " from " + str(Hotel['city'].nunique())+ " cities. ")
Hotel.isnull().sum()
# Perform some basic cleaning and character removal

# Make everything lower case
data['reviews.text'] = data['reviews.text'].str.lower()

# Remove non-text characters
data['reviews.text'] = data['reviews.text'].str.replace(r'\.|\!|\?|\'|,|-|\(|\)', "",)

# Fill in black reviews with '' rather than Null (which would give us errors)
data['reviews.text'] = data['reviews.text'].fillna('')
data['reviews.text'][:5]
# Import and initiate a vectorizer
from sklearn.feature_extraction.text import CountVectorizer
# The max features is how many words we want to allow us to create columns for
vectorizer = CountVectorizer(max_features=5000)
# Vectorize our reviews to transform sentences into volumns
X = vectorizer.fit_transform(data['reviews.text'])

# And then put all of that in a table
bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
# Rename some columns for clarity
data.rename(columns={'address': 'hotel_address', 'city': 'hotel_city',
                     'country':'hotel_country', 'name':'hotel_name'},
            inplace=True)

# Join our bag of words back to our initial hotel data
full_df = data.join(bag_of_words)
# X is our words
X = bag_of_words

# Y is our hotel name (the outcome we care about)
Y_hotel = data['hotel_name']
# Import a random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# Fit that random forest model to our data
rfc.fit(X,Y_hotel)
# Write your own dream vacation review here...
test_review = ['''
    I loved the beach and the sunshine and the clean and modern room.
    ''']
# Convert your test review into a vector
X_test = vectorizer.transform(test_review).toarray()
 #Match your review
prediction = rfc.predict(X_test)[0]
# Return the essential information about your match
data[data['hotel_name'] == prediction][['hotel_name', 'hotel_address', 
                                        'hotel_city', 'hotel_country']].head(1)