# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error 

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the train data 

train_file_path = '../input/learn-together/train.csv'

train_data = pd.read_csv(train_file_path)

train_data.head()
#Load the test data

test_file_path = '../input/learn-together/test.csv'

test_data = pd.read_csv(test_file_path)

test_data.head()
# parameter considered in measurements

train_data.columns
# Ploting any feature from the train and test data

fig, axs = plt.subplots(figsize=(15,4))



feature = 'Slope'

sns.distplot(train_data[feature], label = 'train')

plt.legend()

plt.title(feature)

sns.distplot(test_data[feature], label = 'test')

plt.legend()

plt.title(feature)

plt.show()
# use the Pearson correlation to have a global view of data and its interdependence

cor=train_data.corr()



plt.figure(figsize=(50,35))

plt.title('Pearson Corr of Features', y=1.05, size=15)

sns.heatmap(cor,  square=True,  linecolor='white', annot=True ,  cmap="pink")
#High correlation (relevant) feature selection

cor_target = abs(cor["Cover_Type"])

relevant_features = cor_target[cor_target>0.05]

print(relevant_features)



# get relevant fetures in an array

features = []

for col in relevant_features.index:

    features.append(col)

print(features)
# The Pearsons correlation of relevant features, just to have a zoom-like view 

train_features = train_data[features]

corT=train_features.corr()

plt.figure(figsize=(50,35))

sns.heatmap(corT,  square=True,  linecolor='white', annot=True ,  cmap=plt.cm.RdBu)

features.remove('Cover_Type') # we need this for the scope variable (y) in train,

                              # the rest is the fauteres considered (X) 
# defining the variables for the model

y = train_data.Cover_Type

X = train_data[features]

X.head()
# taking randomly the train and test parts of data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

# the model

model = RandomForestClassifier(n_estimators=150, max_depth=50)

# fitting 

model.fit(X_train,y_train)

# testing

predict = model.predict(X_val)

print(mean_absolute_error(y_val, predict))

print(model.score(X_train,y_train))

print(model.score(X_val,y_val))



predict_test = model.predict(test_data[features])

predict_test
output = pd.DataFrame({'Id': test_data['Id'], 'Cover_Type': predict_test})

output.to_csv('submission.csv', index=False)