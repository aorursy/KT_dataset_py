import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))
# Set our train and test date

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
# data size

train_df.shape
# Set features and label for showing

digits = train_df.drop(['label'], 1).values

digits = digits / 255

label = train_df['label'].values
# Show 25 digits of data

fig, axis = plt.subplots(5, 5, figsize=(22, 20))



for i, ax in enumerate(axis.flat):

    ax.imshow(digits[i].reshape(28, 28), cmap='binary')

    ax.set(title = "Real digit is {}".format(label[i]))
# Machine Learning

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
digits.shape
# Set X, y for fiting

X = digits

y = label

#X_test = test_df.values # file data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier



# Seting our model

model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test) # predict our file test data
print("Model accuracy is: {0:.3f}%".format(accuracy_score(y_test, y_pred) * 100))
# Compare our result

fig, axis = plt.subplots(5, 5, figsize=(18, 20))



for i, ax in enumerate(axis.flat):

    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')

    ax.set(title = "Predicted digit {0}\nTrue digit {1}".format(y_pred[i], y_test[i]))
test_X = test_df.values

rfc_pred = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
# Make submission file

sub['Label'] = rfc_pred

sub.to_csv('submission.csv', index=False)
# Show our submission file

sub.head(10)