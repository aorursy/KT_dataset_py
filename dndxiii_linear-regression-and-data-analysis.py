# Import stuff



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, mean_squared_error

%matplotlib inline

# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))
df = pd.read_csv('../input/StudentsPerformance.csv')

df.head()
df.info()
df.describe()
df["average score"] = (df["math score"] + df["reading score"] + df["writing score"]) /3

df['average score group'] = pd.cut(df["average score"], bins=[g for g in range(0, 101, 10)], include_lowest=True)

df.hist(bins=20, figsize=(12,8))
plt.figure(figsize=(12, 8))

p = sns.countplot(x='parental level of education', data = df, palette='deep')
plt.figure(figsize=(12, 8))





p = sns.countplot(x='parental level of education', data = df, hue='average score group', palette="deep")
plt.figure(figsize=(12, 8))

p = sns.countplot(x='race/ethnicity', data = df, palette='deep')
plt.figure(figsize=(12, 8))





p = sns.countplot(x='race/ethnicity', data = df, hue='average score group', palette="deep")
plt.figure(figsize=(12, 8))

p = sns.countplot(x='lunch', data = df, palette='deep')
plt.figure(figsize=(12, 8))





p = sns.countplot(x='lunch', data = df, hue='average score group', palette="deep")
fr_lunch = df[df['lunch']=='free/reduced']

std_lunch = df[df['lunch']=='standard']



print("Free/Reduced lunch mean",fr_lunch['average score'].mean())

print("Standard lunch mean",std_lunch['average score'].mean())
new_df = df.copy()



one_hot = pd.get_dummies(df['gender'], prefix='gender')

new_df = new_df.join(one_hot)

one_hot = pd.get_dummies(df['race/ethnicity'], prefix='race/ethnicity')

new_df = new_df.join(one_hot)

one_hot = pd.get_dummies(df['parental level of education'], prefix='parental level of education')

new_df = new_df.join(one_hot)

one_hot = pd.get_dummies(df['lunch'], prefix='lunch')

new_df = new_df.join(one_hot)

one_hot = pd.get_dummies(df['test preparation course'], prefix='test preparation course')

new_df = new_df.join(one_hot)



new_df.drop(["reading score", "writing score", "math score", "gender", "race/ethnicity", "parental level of education", "test preparation course","lunch", "average score group"], axis=1, inplace=True)



new_df.head()
train_set, test_set = train_test_split(new_df, test_size=0.20, random_state=21)



train_X = train_set.drop('average score', axis=1)

train_Y = train_set['average score'].copy()



test_X = test_set.drop('average score', axis=1)

test_Y = test_set['average score'].copy()


lin_reg = LinearRegression()



results = cross_validate(lin_reg, train_X, train_Y, cv=10, return_estimator=True)



scores = results['test_score']

print("Scores:",scores)

print("Mean:", scores.mean())

print("Standard deviation:", scores.std())
# Find the best model



best = np.where(scores == min(scores))[0][0]

best_estimator = results['estimator'][best]

final_predictions = best_estimator.predict(test_X)

final_mse = mean_squared_error(test_Y, final_predictions)

final_rmse = np.sqrt(final_mse)

print(final_rmse)


