# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import csv as csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

%matplotlib inline
train_data_origin = []

test_data_origin = []



# load train data

with open('../input/train.csv', 'rt') as train_file:

    reader = csv.reader(train_file)

    for row in reader:

        train_data_origin.append(row)

        

# load test data        

with open('../input/test.csv', 'rt') as test_file:

    reader = csv.reader(test_file)

    for row in reader:

        test_data_origin.append(row)

        

train_data_origin = np.array(train_data_origin[1:])

test_data_origin = np.array(test_data_origin[1:])



print('Num of train data: ' + str(len(train_data_origin)))

print('Num of test data:' + str(len(test_data_origin)))

def mk_float(s):

    s = s.strip()

    return float(s) if s else 0



def mk_int(s):

    s = s.strip()

    return int(s) if s else 0



def mk_sex(s):

    return 1 if s == 'male' else 0
train_data = np.zeros((train_data_origin.shape[0], 5))

test_data = np.zeros((test_data_origin.shape[0], 4))
print(train_data.shape)

print(test_data.shape)
# Survived at column 0

train_data[:, 0] = list(map(mk_int, train_data_origin[:, 1]))

# age at column 1

train_data[:, 1] = list(map(mk_float, train_data_origin[:, 5]))

# fare at column 2

train_data[:, 2] = list(map(mk_float, train_data_origin[:, 9]))

# class - 3

train_data[:, 3] = list(map(mk_int, train_data_origin[:, 2]))

# sex - male = 1

train_data[:, 4] = list(map(mk_sex, train_data_origin[:, 4]))
# age at column 1

test_data[:, 0] = list(map(mk_float, test_data_origin[:, 4]))

# fare at column 2

test_data[:, 1] = list(map(mk_float, test_data_origin[:, 8]))

# class - 3

test_data[:, 2] = list(map(mk_int, test_data_origin[:, 1]))

# sex - male = 1

test_data[:, 3] = list(map(mk_sex, test_data_origin[:, 3]))
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

plt.ylim(0, 300)

plt.scatter(train_data[:, 1], train_data[:, 2], c=train_data[:, 0], s=20, cmap=plt.cm.Spectral)
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

plt.ylim(0, 5)

plt.scatter(train_data[:, 1], train_data[:, 3], c=train_data[:, 0], s=20, cmap=plt.cm.Spectral)

# red = 0, blue = 1
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

plt.ylim(-1, 2)

plt.scatter(train_data[:, 1], train_data[:, 4], c=train_data[:, 0], s=20, cmap=plt.cm.Spectral)

# red = 0, blue = 1
num_survived = np.sum(train_data[:,0].astype(np.float))
survived = train_data[:, 0] == 1
# total survided

np.size(train_data[survived, 4])
# total survived man

np.sum(train_data[survived, 4])
# total man

np.sum(train_data[:,4])
#train_data[survived].shape

#train_data[survived].shape

survived.shape
train_data = pd.read_csv("../input/train.csv")

train_data.describe()
train_data.info()
train_data["Survived"].value_counts()
train_data.hist(bins=50, figsize=(20,20))
corr_matrix = train_data.corr()
corr_matrix['Survived'].sort_values(ascending=False)
attributes=['Survived', 'Fare', 'Age', 'Pclass', 'Parch', 'SibSp']
from pandas.tools.plotting import scatter_matrix
scatter_matrix(train_data[attributes], figsize=(12,8))
train_data.plot(kind="scatter", x="Fare", y="Survived", alpha=0.1)
# Prepare data for Machine Learning

titanic = train_data.drop('Survived', axis = 1)

titanic_labels = train_data['Survived'].copy()
titanic.info()
titanic.head()
median_age = titanic['Age'].median()
median_age
titanic['Age'].fillna(median_age, inplace=True)

titanic = titanic.drop('Cabin', axis=1)
titanic.info()