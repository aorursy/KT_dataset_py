# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv('../input/another-fiat-500-dataset-1538-rows/automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv')
data.head()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data2 = data.copy()
data2 = pd.get_dummies(data)

X = data2.drop(['price'], axis=1)
Y = data2.price

len(X), len(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)

#n_estimators: between 10 and 100
#criterion: mae or mse
#max_depth
#min_sample_split

criterion = ['mae','mse']

best_mae = 0

config = {}

for n_est in range(10,100,5):
    for crit in criterion:
        for depth in range(2,30):
            for m_s_s in range(1,10,1):
                forest_model = RandomForestRegressor(n_estimators=n_est, criterion=crit, max_depth=depth, min_samples_split= m_s_s/10, random_state=1)
                forest_model.fit(X_train, Y_train)
                Y_test_pred = forest_model.predict(X_test)
                candidate_mae = mean_absolute_error(Y_test, Y_test_pred)
                if best_mae == 0 or candidate_mae < best_mae:
                    best_mae = candidate_mae
                    config['n_estimators'] = n_est
                    config['criterion'] = crit
                    config['max_depth'] = depth
                    config['min_sample_split'] = m_s_s/10
                    print("best mae now is " + str(best_mae))
                    print(config)
print('Analysis done!')
#best mae now is 581.9647907647908
#{'n_estimators': 30, 'criterion': 'mae', 'max_depth': 7, 'min_sample_split': 0.1}

forest_model = RandomForestRegressor(n_estimators=30, criterion='mae', max_depth=7, min_samples_split=0.1, random_state=1)
forest_model.fit(X_train, Y_train)
Y_test_pred = forest_model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_test_pred)
print('mae:' + str(mae))
Y_train_pred = forest_model.predict(X_train) 
Y_test_pred = forest_model.predict(X_test)
mae = mean_absolute_error(Y_train, Y_train_pred)
print('mae for train set:' + str(mae))
mae = mean_absolute_error(Y_test, Y_test_pred)
print('mae for test set:' + str(mae))
i = 0
for element in Y_test:
    print('Target is %5d while prediction is %5d (difference: %4d)' %(element, Y_test_pred[i], abs(element - Y_test_pred[i])))
    i = i + 1
best_difference = 10000
worst_difference = 0
i = 0

for target in Y_test:
    predicted_target = Y_test_pred[i]
    difference = abs(target - predicted_target)
    i = i + 1
    if best_difference == 10000 or difference < best_difference:
        best_difference = int(difference)
        #print('best difference:' + str(best_difference))
    if worst_difference == 0 or difference > worst_difference:
        worst_difference = int(difference)
        #print('wort_difference:' + str(worst_difference))
print('Best difference: %d, Worst difference: %d' %(best_difference, worst_difference))
# plot the difference scatter vs the target
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(Y_test_pred, Y_test - Y_test_pred)
plt.grid(True)
plt.xlabel('Target')
plt.ylabel('Difference in prediction')
plt.show()