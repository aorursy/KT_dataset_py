import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
''' MAIN SCRIPT:

In here all stuff should happen. Things above should be callable only

https://www.dataquest.io/blog/sci-kit-learn-tutorial/

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


googleplaystore = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
googleplaystore_user_reviews = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

print(googleplaystore.tail())

# SUBSTITUTE TEXT WITH NUMBERS
le = preprocessing.LabelEncoder()
for col in googleplaystore.columns:
    if type(googleplaystore[col][0]) is str:
        googleplaystore[col] = le.fit_transform(googleplaystore[col])

print(googleplaystore.head())

# PARTITION DATAFRAME
validation_excluded, validation = train_test_split(googleplaystore, test_size=0.2, random_state=42)
train, test = train_test_split(validation_excluded)

print(train.head())

# DROP NAN
train = train.dropna()
test = test.dropna()

print(train.head())

# TRAIN MODEL
cols = [1,5,6]

train_data = train[cols]
train_target = train[2]

data = test[cols]
target = test[2]

# SkLearn SGD classifier
n_iter = 10000
clf_ = SGDRegressor(max_iter=n_iter)
clf_.fit(train_data, train_target)
target_pred = clf_.predict(data)

plt.scatter(target, target_pred)
plt.grid()
plt.xlabel('Actual target data')
plt.ylabel('Predicted target data')
plt.title('Scatter plot from actual data and predicted data')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.show()

print('Mean Squared Error :', mean_squared_error(target, target_pred))