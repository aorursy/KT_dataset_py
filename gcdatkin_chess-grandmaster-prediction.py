import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression



from sklearn.metrics import f1_score
data = pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
data
data.drop(['Fide id', 'Name', 'Gender'], axis=1, inplace=True)
data
data.isnull().sum()
data.dtypes
numerical_features = ['Year_of_birth', 'Rapid_rating', 'Blitz_rating']
for column in numerical_features:

    data[column] = data[column].fillna(data[column].mean())
data.isnull().sum()
data['Title'].unique()
data['Inactive_flag'].unique()
data['Inactive_flag'] = data['Inactive_flag'].fillna('wa')
data.isnull().sum()
title_dummies = pd.get_dummies(data['Title'])

title_dummies
data = pd.concat([data, title_dummies['GM']], axis=1)

data.drop('Title', axis=1, inplace=True)
data
data.isnull().sum()
data['Inactive_flag'].unique()
encoder = LabelEncoder()



data['Inactive_flag'] = encoder.fit_transform(data['Inactive_flag'])
data
data['Federation'].unique()
data.drop('Federation', axis=1, inplace=True)
data
y = data['GM']

X = data.drop('GM', axis=1)
X
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = LogisticRegression()

model.fit(X_train, y_train)
print(f"Model Accuracy: {model.score(X_test, y_test)}")
y_pred = model.predict(X_test)
print(f"Model F1 Score: {f1_score(y_test, y_pred)}")
print(f"Percent Grandmaster: {y_test.sum() / len(y)}")