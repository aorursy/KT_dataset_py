# essentials for getting the input files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# importing the essentials libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# looks like we have only one dataset. 

# importing the dataset into the code

df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
# checking the dimensions of dataset

df.shape
df.dtypes
df.describe()
df.corr(method = 'pearson')
sns.heatmap(df.corr())
# removing missing values

df.isnull().sum()
df = df.drop(["company", "agent"], axis = 1)
df = df.dropna()
df.isnull().sum()
df.head()
df.shape
# importing the label encoder from sklearn

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

def encoding(df_column):

    if (df_column.dtype == 'O'):

        return encoder.fit_transform(df_column)

    return df_column

df_encoded = df.apply(encoding)
df_encoded.head()
df_encoded = df_encoded.drop(['lead_time', 'arrival_date_year', 'reservation_status_date', 'adr'], axis = 1)
df_encoded.corr(method = 'pearson')['is_repeated_guest']
input_features = df_encoded.drop(["is_repeated_guest"], axis = 1)

target_feature = df_encoded[["is_repeated_guest"]]
input_features.info(), target_feature.info()
input_features.describe()
# importing the SelectKBest feature selection

from sklearn.feature_selection import SelectKBest, chi2

best_features = SelectKBest(score_func = chi2, k = 'all')

new_best = best_features.fit(input_features, target_feature)
scores_df = pd.DataFrame(new_best.scores_)

columns_df = pd.DataFrame(input_features.columns)

final_dataframe = pd.concat([columns_df, scores_df], axis = 1)

final_dataframe.columns = ['features', 'scores']

final_dataframe
# sorting them in descending order

final_dataframe.sort_values(by = 'scores', ascending = False)
input_features = df_encoded.drop(['arrival_date_month', 

                                  'customer_type', 

                                  'babies', 

                                  'total_of_special_requests', 

                                  'arrival_date_day_of_month', 

                                  'booking_changes',

                                  'is_repeated_guest'], axis = 1)

target_feature = df_encoded[["is_repeated_guest"]]
input_features.info()
# converting the children feature from float to int

print(input_features['children'].dtype)

input_features['children'] = pd.to_numeric(input_features['children'].values, downcast = 'integer')

print(input_features['children'].dtype)
input_features.info(), target_feature.info()
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(input_features, target_feature, test_size = 0.2, random_state = 101)
print(train_X.shape)

print(train_y.shape)

print(test_X.shape)

print(test_y.shape)
# let us use KNN classification

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = int(np.sqrt(input_features.shape[0]).round()))
classifier.fit(train_X, train_y.values)
predicted_classes = classifier.predict(test_X)
# let us check the accuracy of the model

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(confusion_matrix(test_y, predicted_classes))
(22987 + 131) / (22987 + 21 + 641 + 131)
print(classification_report(test_y, predicted_classes))
print(accuracy_score(test_y, predicted_classes))