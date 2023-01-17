import pandas as pd
cab_df = pd.read_csv("../input/cab_rides.csv",delimiter='\t',encoding = "utf-16")

weather_df = pd.read_csv("../input/weather.csv",delimiter='\t',encoding = "utf-16")
cab_df.head()
weather_df.head()
cab_df['date_time'] = pd.to_datetime(cab_df['time_stamp']/1000, unit='s')

weather_df['date_time'] = pd.to_datetime(weather_df['time_stamp'], unit='s')

cab_df.head()
#merge the datasets to refelect same time for a location

cab_df['merge_date'] = cab_df.source.astype(str) +" - "+ cab_df.date_time.dt.date.astype("str") +" - "+ cab_df.date_time.dt.hour.astype("str")

weather_df['merge_date'] = weather_df.location.astype(str) +" - "+ weather_df.date_time.dt.date.astype("str") +" - "+ weather_df.date_time.dt.hour.astype("str")
weather_df.index = weather_df['merge_date']
cab_df.head()
merged_df = cab_df.join(weather_df,on=['merge_date'],rsuffix ='_w')
merged_df['rain'].fillna(0,inplace=True)
merged_df = merged_df[pd.notnull(merged_df['date_time_w'])]
merged_df = merged_df[pd.notnull(merged_df['price'])]
merged_df['day'] = merged_df.date_time.dt.dayofweek
merged_df['hour'] = merged_df.date_time.dt.hour
merged_df['day'].describe()
merged_df.columns
merged_df.count()
X = merged_df[merged_df.product_id=='lyft_line'][['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain']]
X.count()
y = merged_df[merged_df.product_id=='lyft_line']['price'] 
y.count()
X.reset_index(inplace=True)

X = X.drop(columns=['index'])
X.head()
features = pd.get_dummies(X)
features.columns
# Use numpy to convert to arrays

import numpy as np

# Labels are the values we want to predict

labels = np.array(y)



# Saving feature names for later use

feature_list = list(features.columns)

# Convert to numpy array

features = np.array(features)
# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
# Import the model we are using

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / test_labels)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
merged_df_surge = merged_df[merged_df.surge_multiplier < 3]

X = merged_df_surge[['day','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain']]
X.count()
features = pd.get_dummies(X)


y = merged_df_surge['surge_multiplier']

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



#ignoring multiplier of 3 as there are only 2 values in our dataset

le.fit([1,1.25,1.5,1.75,2.,2.25,2.5])

y = le.transform(y) 
# Use numpy to convert to arrays

import numpy as np

# Labels are the values we want to predict

labels = np.array(y)



# Saving feature names for later use

feature_list = list(X.columns)

# Convert to numpy array

features = np.array(features)
# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

train_features, train_labels = sm.fit_resample(train_features, train_labels)
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state = 42,class_weight="balanced")

# Train the model on training data

rf.fit(train_features, train_labels);
# Use the forest's predict method on the test data

predictions = rf.predict(test_features)

# Calculate the absolute errors

errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
from sklearn.metrics import precision_score, recall_score

precision_score(test_labels, predictions, average="weighted")

recall_score(test_labels, predictions, average="micro")
# Create confusion matrix

pd.crosstab(le.inverse_transform(test_labels), le.inverse_transform(predictions),rownames=['Actual'],colnames=['Predicted'])
# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, predictions)