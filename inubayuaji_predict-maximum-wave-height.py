# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data

path = '../input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'

data = pd.read_csv(path)
# sample data

data.head()
# data shape

data.shape
# show nan value

data.isna().sum()
# there anomaly data -99.9 taht need delete

data = data[data['Hmax'] != -99.9]
# corelation data

sns.heatmap(data.corr(), annot=True)
# chose future and target variable

x = ['Hs', 'Tz', 'Tp', 'Peak Direction', 'SST']

y = ['Hmax']



x_train, x_test, y_train, y_test = train_test_split(data[x], data[y],

                                                    test_size=0.3,

                                                    shuffle = False)
# make model

xgb = XGBRegressor(learning_rate=0.06, max_depth=3, n_estimators=100)
# train model

xgb.fit(x_train, y_train)
#predict

result = xgb.predict(x_test)
# mae result

mae = mean_absolute_error(result, y_test)

print(mae)
y_test['Hmax'].index.values
# plot data y_train, y_test, and result

plt.figure(figsize=(12,5), dpi=100)

plt.plot(y_train['Hmax'], color='blue')

plt.plot(y_test, color='yellow')

plt.plot(y_test.index.values, result, color='green')

plt.show()