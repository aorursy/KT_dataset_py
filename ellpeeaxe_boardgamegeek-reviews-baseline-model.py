# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
detail_df = pd.read_csv('/kaggle/input/boardgamegeek-reviews/games_detailed_info.csv', index_col=0)

detail_df.head()
updated_detail_df = detail_df.copy()



updated_detail_df = updated_detail_df[

    ['averageweight', 'bayesaverage', 'maxplayers', 'maxplaytime', 'minage', 

     'minplayers', 'minplaytime','playingtime']]

updated_detail_df.head()



# Features

x = updated_detail_df[['averageweight', 'maxplayers', 'maxplaytime', 'minage', 

     'minplayers', 'minplaytime','playingtime']]

# Target

y = updated_detail_df['bayesaverage']
# Split into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model

model = LinearRegression()  

model.fit(x_train, y_train)
# View model coefficients

coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])  

coeff_df
# Compare results

predictions = model.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

df.head(25)
# Calculate errors

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
threshold = 5.702  # 50th percentile 5.556, 75th percentile 5.702



# Change target to binary

y_bin = y.map(lambda x: 0 if x<threshold else 1)
# Split into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y_bin, test_size=0.2, random_state=0)
# Train model

model = LogisticRegression()  

model.fit(x_train, y_train)
# Compare results

predictions = model.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

df.head(25)
# Evaluate

score = model.score(x_test, y_test)

score
# Precision and Recall

precision = metrics.precision_score(y_test, predictions)

recall = metrics.recall_score(y_test, predictions)

f_measure = metrics.f1_score(y_test, predictions)

print(precision)

print(recall)

print(f_measure)