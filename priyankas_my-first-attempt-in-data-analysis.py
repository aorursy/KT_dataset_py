import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

print(train_data.shape)

print(test_data.shape)

print(train_data.columns)

ob_col= (train_data.select_dtypes(include=['object'])).columns

print(ob_col)

print(train_data.describe())
for col in train_data.columns:

    if len(train_data[col].unique()) == 1:

        del train_data[col]

print(train_data.columns)
y = train_data['y']

X = train_data.drop(['y'], axis=1)



test_data_refined = test_data[X.columns]

print(X.columns)

print(test_data_refined.columns)
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(8, 1, figsize=(10, 30))

i=0

for c in ob_col:

    X[c].value_counts().sort_index().plot.bar(

    ax=axarr[i], fontsize=12, color='mediumvioletred')

    axarr[i].set_title(c, fontsize=12)

    i=i+1
train_data_onehot = pd.get_dummies(X)

test_data_onehot = pd.get_dummies(test_data_refined)

train_predictors,test_predictors = train_data_onehot.align(test_data_onehot,join='inner',axis=1)



print(train_predictors.shape)

print(test_predictors.shape)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(train_predictors,y)

predictions = model.predict(test_predictors)
ans_df = pd.DataFrame({'ID': test_data.ID, 'y': predictions})

print(ans_df.describe())

ans_df.to_csv('answer.csv', index=False)