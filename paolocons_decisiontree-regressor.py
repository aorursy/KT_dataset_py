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
data = pd.read_csv('../input/small-dataset-about-used-fiat-500-sold-in-italy/Used_fiat_500_in_Italy_dataset.csv', encoding='utf-8')

data.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



data2 = data.copy()

data2 = pd.get_dummies(data)

data2.head()

X = data2.drop(['price'], axis=1)

Y = data2.price
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=1)
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
#best mae now is 483.03815789473686

#{'n_estimators': 30, 'criterion': 'mae', 'max_depth': 9, 'min_sample_split': 0.1}



forest_model = RandomForestRegressor(n_estimators=30, criterion='mae', max_depth=9, min_samples_split=0.1, random_state=1)



forest_model.fit(X_train, Y_train)



Y_train_pred = forest_model.predict(X_train)

mae = mean_absolute_error(Y_train, Y_train_pred)

print('mae on train set: %.2f' %(mae))



Y_test_pred = forest_model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_test_pred)

print('mae on test  set: %.2f' %(mae))
# print differences between real and predicted on train features

i = 0



predictions = forest_model.predict(X_train)

worst_difference = 0

best_difference = 10000



for item in Y_train.iteritems():

    real_price = item[1]

    predicted_price  = predictions[i]

    difference = real_price - predicted_price

    print('Real price is %5d, predicted is %5d (difference: %5d)' %(real_price, predicted_price, difference))

    if worst_difference == 0 or abs(difference) > worst_difference:

        worst_difference = abs(difference)

    if best_difference == 10000 or abs(difference) < best_difference:

        best_difference = abs(difference)

    i = i + 1



print('\nAbsolute worst difference was %5d'  %(worst_difference))

print('\nAbsolute best difference was %5d'   %(best_difference)) 
# print differences between real and predicted on test features

i = 0



predictions = forest_model.predict(X_test)

worst_difference = 0

best_difference = 10000



for item in Y_test.iteritems():

    real_price = item[1]

    predicted_price  = predictions[i]

    difference = real_price - predicted_price

    print('Real price is %5d, predicted is %5d (difference: %5d)' %(real_price, predicted_price, difference))

    if worst_difference == 0 or abs(difference) > worst_difference:

        worst_difference = abs(difference)

    if best_difference == 10000 or abs(difference) < best_difference:

        best_difference = abs(difference)

    i = i + 1



print('\nAbsolute worst difference was %5d'  %(worst_difference))

print('\nAbsolute best difference was %5d'   %(best_difference)) 