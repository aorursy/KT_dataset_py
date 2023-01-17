import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
print(df_train.head())
print(df_test.head())
df_train.info()
df_test.info()
print("No. of rows in train set: ", len(df_train))
print("No. of rows in test set: ", len(df_test))
df_train = df_train.dropna(axis=0)
df_test = df_test.dropna(axis=0)
print("No. of rows in train set: ", len(df_train))
print("No. of rows in test set: ", len(df_test))
# features = df_train.columns.drop(["winPlacePerc", "Id", "groupId", "matchId"])
features = ['walkDistance', 'killPlace', 'boosts', 'weaponsAcquired']
train_X = df_train[features]
train_y = df_train['winPlacePerc']
test_X = df_test[features]

#one hot encode
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(train_X, train_y)
predict_y = forest_model.predict(test_X)
predict_y
output = pd.DataFrame({'Id': df_test.Id,
                       'winPlacePerc': predict_y})

output.to_csv('submission.csv', index=False)
output