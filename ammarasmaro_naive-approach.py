import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
os.listdir('../input')
data_file = os.path.join('..','input', 'FIFA 2018 Statistics.csv')
dataframe = pd.read_csv(data_file)
dataframe.describe()
dataframe.head()
dataframe.columns
dataframe.corr()
def normalize(column):
    mean = column.mean()
    std = column.std()
    return column.apply(lambda x: (x - mean) / std)
numeric_cols = ['Goal Scored', 'Ball Possession %', 'Attempts', 'On-Target', 'Off-Target', 'Blocked', 'Corners', 'Offsides',
       'Free Kicks', 'Saves', 'Pass Accuracy %', 'Passes',
       'Distance Covered (Kms)', 'Fouls Committed', 'Yellow Card',
       'Yellow & Red', 'Red', '1st Goal',
       'Goals in PSO', 'Own goals', 'Own goal Time']
for num_column in numeric_cols:
    dataframe[num_column] = pd.to_numeric(dataframe[num_column], errors='coerce')
dataframe.fillna(0, inplace=True)
dataframe.describe()
for num_column in numeric_cols:
    dataframe[num_column] = normalize(dataframe[num_column])
dataframe.describe()
categorical_cols = ['Man of the Match']
for categorical_column in categorical_cols:
    dataframe[categorical_column] = dataframe[categorical_column].apply(lambda x: 1 if x == 'Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(dataframe[numeric_cols], dataframe['Man of the Match'],
                                                    test_size=0.33, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)
new_model = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(dataframe[['Goal Scored', '1st Goal', 'On-Target', 'Attempts', 'Corners']], dataframe['Man of the Match'],
                                                    test_size=0.33, random_state=42)
model.fit(X_train, y_train)
model.score(X_test, y_test)
