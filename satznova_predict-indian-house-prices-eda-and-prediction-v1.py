# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/train.csv')

print(train_df.columns)
print(train_df.info())
train_df.head()
print(train_df['POSTED_BY'].unique())
print(train_df['BHK_OR_RK'].unique())

print("--- ---\n\nRecords for POSTED_BY: ")
print(train_df['POSTED_BY'].value_counts())

print("--- ---\n\nRecords for BHK_OR_RK: ")
print(train_df['BHK_OR_RK'].value_counts())

print("--- ---\n\nRecords for READY_TO_MOVE: ")
print(train_df['READY_TO_MOVE'].value_counts())


print("--- ---\n\nRecords for RESALE: ")
print(train_df['RESALE'].value_counts())

print("--- ---\n\nRecords for UNDER_CONSTRUCTION: ")
print(train_df['UNDER_CONSTRUCTION'].value_counts())

print("--- ---\n\nRecords for RERA: ")
print(train_df['RERA'].value_counts())

print("--- ---\n\nRecords for BHK_NO.: ")
print(train_df['BHK_NO.'].value_counts())
tier_1_cities = ['Ahmedabad', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']
tier_2_cities = ['Agra', 'Ajmer', 'Aligarh', 'Amravati', 'Amritsar', 'Asansol', 'Aurangabad', 'Bareilly', 'Belgaum', 'Bhavnagar', 'Bhiwandi', 
                 'Bhopal', 'Bhubaneswar', 'Bikaner', 'Bilaspur', 'Bokaro Steel City', 'Chandigarh', 'Coimbatore', 'Cuttack', 'Dehradun', 'Dhanbad',
                 'Bhilai', 'Durgapur', 'Dindigul', 'Erode', 'Faridabad', 'Firozabad', 'Ghaziabad', 'Gorakhpur', 'Gulbarga', 'Guntur', 'Gwalior', 
                 'Gurgaon', 'Guwahati', 'Hamirpur', 'Hubli–Dharwad', 'Indore', 'Jabalpur', 'Jaipur', 'Jalandhar', 'Jammu', 'Jamnagar', 'Jamshedpur', 
                 'Jhansi', 'Jodhpur', 'Kakinada', 'Kannur', 'Kanpur', 'Karnal', 'Kochi', 'Kolhapur', 'Kollam', 'Kozhikode', 'Kurnool', 'Ludhiana', 
                 'Lucknow', 'Madurai', 'Malappuram', 'Mathura', 'Mangalore', 'Meerut', 'Moradabad', 'Mysore', 'Nagpur', 'Nanded', 'Nashik', 'Nellore',
                 'Noida', 'Patna', 'Pondicherry', 'Purulia', 'Prayagraj', 'Raipur', 'Rajkot', 'Rajahmundry', 'Ranchi', 'Rourkela', 'Salem', 'Sangli', 
                 'Shimla', 'Siliguri', 'Solapur', 'Srinagar', 'Surat', 'Thanjavur', 'Thiruvananthapuram', 'Thrissur', 'Tiruchirappalli', 'Tirunelveli', 
                 'Ujjain', 'Bijapur', 'Vadodara', 'Varanasi', 'Vasai-Virar City', 'Vijayawada', 'Visakhapatnam', 'Vellore', 'Warangal']
train_df['city'] = train_df['ADDRESS'].str.split(",").str[-1]
train_df.head()
train_df.groupby(['city'])['ADDRESS'].count().reset_index(name= 'count').sort_values(['count'], ascending=False).head(10)
def check_city_tier(row, tier_val):
    if tier_val == 1:
        if row['city'] in tier_1_cities:
            return 1
    elif tier_val == 2:
        if row['city'] in tier_2_cities:
            return 1
    elif tier_val == 3:
        if row['city'] not in tier_1_cities and row['city'] not in tier_2_cities:
            return 1
    return 0

train_df['tier_1_city'] = train_df.apply(check_city_tier, args=([1]), axis=1)
train_df['tier_2_city'] = train_df.apply(check_city_tier, args=([2]), axis=1)
train_df['tier_other_cities'] = train_df.apply(check_city_tier, args=([3]), axis=1)

train_df
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[("posted_by_transform", OneHotEncoder(), ['POSTED_BY'])], remainder='passthrough')

train_df_tranformed = pd.DataFrame(transformer.fit_transform(train_df))

# Set Column names in for the transformed Dataframe
train_df_tranformed.columns = ['POSTED_BY_builder', 'POSTED_BY_dealer', 'POSTED_BY_Owner','UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'SQUARE_FT', 
                               'READY_TO_MOVE', 'RESALE', 'ADDRESS', 'LONGITUDE', 'LATITUDE', 'TARGET(PRICE_IN_LACS)', 'city', 'tier_1_city', 'tier_2_city', 
                               'tier_other_cities']

train_df_tranformed
selected_features = ['POSTED_BY_builder', 'POSTED_BY_dealer', 'POSTED_BY_Owner','UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.',
       'SQUARE_FT', 'READY_TO_MOVE', 'RESALE', 'TARGET(PRICE_IN_LACS)', 'tier_1_city',
       'tier_2_city', 'tier_other_cities']

#selected_features = ['BHK_NO.', 'RESALE', 'SQUARE_FT', 'TARGET(PRICE_IN_LACS)']

df_features_selected = train_df_tranformed[selected_features]

for feature in selected_features:
    df_features_selected[feature] =  pd.to_numeric(df_features_selected[feature])
df_features_selected
from sklearn.model_selection import train_test_split

#train, test = train_test_split(df_features_selected,test_size=0.20, random_state=0)
train, test = train_test_split(df_features_selected,test_size=0.20, random_state=0, stratify=df_features_selected[['READY_TO_MOVE', 'RESALE', 'UNDER_CONSTRUCTION', 'RERA']])

X_train = train[train.columns.difference(['TARGET(PRICE_IN_LACS)'])]
y_train = train['TARGET(PRICE_IN_LACS)']

X_test = test[test.columns.difference(['TARGET(PRICE_IN_LACS)'])]
y_test = test['TARGET(PRICE_IN_LACS)']

print(X_train.shape)
print(X_test.shape)
# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train, y_train)

print('KNN Regressor: ', lin_reg_model.score(X_test, y_test))
# KNN Regressor

from sklearn.neighbors import KNeighborsRegressor

knn_reg_model = KNeighborsRegressor(n_neighbors=5)

knn_reg_model.fit(X_train, y_train)

print('KNN Regressor: ', knn_reg_model.score(X_test, y_test))
# Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)

print('Random Forest: ', r2_score(y_test, preds))
# XG Boost

from xgboost import XGBRegressor

xgboost_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)

xgboost_model.fit(X_train, y_train)

preds = xgboost_model.predict(X_test)

print('XG Boost: ', r2_score(y_test, preds))
test_df = pd.read_csv('/kaggle/input/house-price-prediction-challenge/test.csv')

# Feature Engineering
# 1. Derive city tier feature
test_df['city'] = test_df['ADDRESS'].str.split(",").str[-1]
test_df['tier_1_city'] = test_df.apply(check_city_tier, args=([1]), axis=1)
test_df['tier_2_city'] = test_df.apply(check_city_tier, args=([2]), axis=1)
test_df['tier_other_cities'] = test_df.apply(check_city_tier, args=([3]), axis=1)

# 2. One-hot encoding for POSTED_BY
test_df_tranformed = pd.DataFrame(transformer.fit_transform(test_df))

# Set Column names in for the transformed Dataframe
test_df_tranformed.columns = ['POSTED_BY_builder', 'POSTED_BY_dealer', 'POSTED_BY_Owner','UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'SQUARE_FT', 
                               'READY_TO_MOVE', 'RESALE', 'ADDRESS', 'LONGITUDE', 'LATITUDE', 'city', 'tier_1_city', 'tier_2_city', 
                               'tier_other_cities']

# Making Prediction
# Using the trained XG Boost model
selected_features = ['BHK_NO.', 'POSTED_BY_Owner', 'POSTED_BY_builder', 'POSTED_BY_dealer',
       'READY_TO_MOVE', 'RERA', 'RESALE', 'SQUARE_FT', 'UNDER_CONSTRUCTION',
       'tier_1_city', 'tier_2_city', 'tier_other_cities']

X_test= test_df_tranformed[selected_features]
for feature in selected_features:
    X_test[feature] =  pd.to_numeric(X_test[feature])

df_test_output = xgboost_model.predict(X_test)

# Save to submission.csv file
df_output = pd.DataFrame({'Id': X_test.index, 'PredictedSalePrice': df_test_output})
df_output.to_csv('submission.csv', index=False)
df_output.head()
