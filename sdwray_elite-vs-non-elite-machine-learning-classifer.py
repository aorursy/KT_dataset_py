# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import libraries
import  matplotlib.pyplot  as plt
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy import stats
# Read in data
user_data = pd.read_csv("../input/yelp_user.csv")
print(user_data.head(2))
# Created new dataframe based of relevant columns
clean_user_data = user_data[["user_id","elite","name","review_count", "yelping_since"]].copy()
print(clean_user_data.head(2))
# Take out users who nevers reviewed anything
only_reviews_data = clean_user_data[clean_user_data.review_count != 0]
only_reviews_data.head(3)

#Seperate user data between basic and elite users
elite_users = only_reviews_data[only_reviews_data.elite != "None"]
basic_users = only_reviews_data[only_reviews_data.elite == "None"]
elite_users["today"] = "2018-07-24"
basic_users["today"] = "2018-07-24"
elite_users["date_diff"] = pd.to_datetime(elite_users.today) - pd.to_datetime(elite_users.yelping_since)  
basic_users["date_diff"] = pd.to_datetime(basic_users.today) - pd.to_datetime(basic_users.yelping_since)
elite_users.head(5)
basic_users.head(5)
elite_users["name"].value_counts()
basic_users["name"].value_counts()
only_reviews_data.head(3)
#Create a column to label data whether they have been an elite user or not
only_reviews_data['status'] = np.where(only_reviews_data['elite']== "None", 'basic', 'elite')
only_reviews_data.head(3)
#Visualization of the amount of reviews written
plt.scatter(list(elite_users.index.values),elite_users['review_count'],c = "red", alpha = .25)
plt.scatter(list(basic_users.index.values),basic_users['review_count'],c = "blue", alpha = .25)
plt.show()
#Create your labels and label data
labels_names = only_reviews_data["status"].unique()
labels = only_reviews_data["status"]
print(labels_names)
print(labels[0:3])
#Create feature name and the features
feature_name = "review_count"
features = pd.DataFrame(only_reviews_data["review_count"])
print(feature_name)
print(features[0:10])
# Split the data into training and testing sets
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)
# Initialize the classifier from sklearn
gnb = GaussianNB()

# Train our classifier using our data splits
model = gnb.fit(train, train_labels)
# Make predictions and show what the classifier predicted counts
preds = gnb.predict(test)
print(preds)
print(stats.itemfreq(preds))

print(accuracy_score(test_labels, preds))
