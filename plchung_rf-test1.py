# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import csv

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_file = '../input/train.csv'

print('reading training file to np')

dat = np.genfromtxt(train_file, delimiter=',')



training_size = dat.shape[0]

train_end = math.floor(training_size * 0.7)

test_start = train_end + 1



x_train = dat[1:train_end, 1:]

y_train = dat[1:train_end, 0]



print('training...')

forest = RandomForestClassifier(n_estimators=250, max_features=40)



fitted_forest = forest.fit(x_train, y_train)

print('trained RF')



x_cv = dat[test_start:, 1:]

y_cv = dat[test_start:, 0]

print('Predicting on CV')

output = fitted_forest.predict(x_cv)



matched = np.sum(output == y_cv)



ratio = float(matched) / len(y_cv)

print('CV result', ratio)



test_dat = np.genfromtxt('../input/test.csv', delimiter=',')



x_test = test_dat[1:, :]

print('Predicting on testing')

y_test = fitted_forest.predict(x_test)



print('Writing prediction output csv')

prediction_file = open('./prediction.csv', 'w', newline='\n', encoding='utf-8')

prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(['ImageId', 'Label'])



for idx, val in enumerate(y_test):

    prediction_file_object.writerow([idx + 1, int(val)])



prediction_file.close()



print('done')
