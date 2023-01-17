#Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import os
import json
from pandas.io.json import json_normalize
#Define function to load data (Kudos to Julian for that)
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
print('Loading Train set...')
train = load_df()
print('Loading Test set...')
test = load_df("../input/test.csv")

#Impute missing values (mean for numeric, mode for categorical)
print('Imputing missing values...')
for i in train.columns:
    if train[i].dtype == 'object':
      train[i] = train[i].fillna(train[i].mode().iloc[0])
    elif (train[i].dtype != 'object'):
      train[i] = train[i].fillna(np.mean(train[i]))


for i in test.columns:
    if test[i].dtype == 'object':
      test[i] = test[i].fillna(test[i].mode().iloc[0])
    elif (test[i].dtype != 'object'):
      test[i] = test[i].fillna(np.mean(test[i]))
    

print('Nulls in train set:', train.isnull().sum().sum())
print('Nulls in test set:', test.isnull().sum().sum())
## label encode categorical variables
print('Label Encoding categorical variables...')
for col in train.columns:
    if train[col].dtype == 'object':
      train[col] = train[col].astype('category')
      train[col] = train[col].cat.codes

for col in test.columns:
    if test[col].dtype == 'object':
      test[col] = test[col].astype('category')
      test[col] = test[col].cat.codes
## Creating a dummy y label and drop the target variable
train['set'] = 0
test['set'] = 1
train = train.drop(['totals.transactionRevenue', 'trafficSource.campaignCode'],axis=1)
## Use a sample set from both train and test and concatenate into a single dataframe
train_df = train.sample(10000, random_state=697)
test_df = test.sample(10000, random_state=466)

all_data = train_df.append(test_df)
y_label = all_data['set']
all_data = all_data.drop('set',axis=1)

#Make sure the new dataframe contains all the initial features
print('New dataframe shape:', all_data.shape)
## Find all the features with covariate shift. Print during the procedure and then save in array
model = RandomForestClassifier(n_estimators = 50, max_depth = 5, min_samples_leaf = 5)
feat_to_drop = []
for col in all_data.columns:
    score = cross_val_score(model,pd.DataFrame(all_data[col]),y_label,cv=4,scoring='roc_auc')
    if np.mean(score) > 0.8:
        feat_to_drop.append(col)
    print(col,np.mean(score))
#Print number of features with covariate shift
print('Number of features with covariate shift:', len(feat_to_drop))