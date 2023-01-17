import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv")

data.head()
data.dtypes
data = data.drop("full_name", axis=1)

data["jersey"] = data["jersey"].str[1:].astype("int8")

data["b_day"] = pd.to_datetime(data["b_day"]).dt.year

data["height"] = data["height"].str.split("/").str[1].astype("float")

data["weight"] = data["weight"].str.split("/").str[1].str[0:-3].astype("float")

data["salary"] = data["salary"].str[1:].astype("int64")

data["draft_round"] = data["draft_round"].replace({"Undrafted": 0}).astype("int8")

data["draft_peak"] = data["draft_peak"].replace({"Undrafted": 0}).astype("int8")

data
data.dtypes
data.isnull().sum()
data['team'] = data['team'].fillna('No team')

data['college'] = data['college'].fillna('No college')
for column in ['team', 'position', 'country', 'college']:

    encoded_columns = pd.get_dummies(data[column], prefix=column)

    data = data.join(encoded_columns).drop(column, axis=1)
y = data["salary"]

X = data.drop("salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = preprocessing.StandardScaler().fit(X_train)  

X_train_normalized = scaler.transform(X_train)      

X_test_normalized = scaler.transform(X_test)    
model_RF = RandomForestRegressor(random_state=7)

params_RF = {

    "n_estimators": [200, 150] ,

    "max_depth": [15, 10],

    "min_samples_split": [2, 4, 8],

    "max_features": ["sqrt", "log2"]

}

model_RF = GridSearchCV(model_RF, params_RF, scoring="neg_mean_squared_error" )

model_RF.fit(X_train_normalized, np.log(y_train))
model_RF.best_params_
model_RF.best_score_
y_pred_RF = model_RF.predict(X_test_normalized)

mse = mean_squared_error(np.log(y_test), y_pred_RF)

print("Test mean squered error:", mse)