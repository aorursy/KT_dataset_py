import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')
eye_data = pd.read_csv(r"../input/eye movement.csv")
eye_data.info()
eye_data.describe()
eye_data.head()
fig,ax = plt.subplots(nrows = 4, ncols=4, figsize=(16,10))

row = 0

col = 0

for i in range(len(eye_data.columns) -1):

    if col > 3:

        row += 1

        col = 0

    axes = ax[row,col]

    sns.boxplot(x = eye_data['Target'], y = eye_data[eye_data.columns[i]],ax = axes)

    col += 1

plt.tight_layout()

# plt.title("Individual Features by Class")

plt.show()
p = eye_data.hist(figsize = (20,20),bins=50)
color_wheel = {1: "#0392cf", 

               2: "#7bc043"}

colors = eye_data["Target"].map(lambda x: color_wheel.get(x + 1))

p = eye_data.Target.value_counts().plot(kind="bar")

plt.xlabel("Target")

plt.ylabel("Count of Target")
X = eye_data.drop(['Target'],axis=1)

y = eye_data.Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=20,shuffle=True)
test_scores = []

train_scores = []

for i in range(1,15):

    

    knn = KNeighborsClassifier(i,weights="distance")

    

    knn.fit(X_train,y_train)

    

    test_scores.append(knn.score(X_test,y_test))

    train_scores.append(knn.score(X_train,y_train))

    
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
from numpy import mean

from numpy import std

from numpy import delete

from numpy import savetxt

# load the dataset.

data = eye_data

values = data.values

# step over each EEG column

for i in range(values.shape[1] - 1):

	# calculate column mean and standard deviation

	data_mean, data_std = mean(values[:,i]), std(values[:,i])

	# define outlier bounds

	cut_off = data_std * 4

	lower, upper = data_mean - cut_off, data_mean + cut_off

	# remove too small

	too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]

	values = delete(values, too_small, 0)

	print('>deleted %d rows' % len(too_small))

	# remove too large

	too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]

	values = delete(values, too_large, 0)

	print('>deleted %d rows' % len(too_large))

# save the results to a new file

savetxt('eye_movement_no_outliers.csv', values, delimiter=',')
eye_data_no_out = pd.read_csv('eye_movement_no_outliers.csv',names = eye_data.columns )

eye_data_no_out.plot(kind='box')
eye_data_no_out.describe()
X = eye_data_no_out.drop(['Target'],axis=1)

y = eye_data_no_out.Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=20,shuffle=True)
from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,7):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,7),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,7),test_scores,marker='o',label='Test Score')