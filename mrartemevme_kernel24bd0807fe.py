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
train = pd.read_csv('/kaggle/input/berlin-airbnb-prices/berlin_airbnb_train.csv')

test = pd.read_csv('/kaggle/input/berlin-airbnb-prices/berlin_airbnb_test.csv')

sample_submission = pd.read_csv('/kaggle/input/berlin-airbnb-prices/submit.csv')

train.head()
# Выбросил все не - числовые признаки

train = train[train.dtypes[(train.dtypes != object)].index]

test = test[test.dtypes[(test.dtypes != object)].index]

from sklearn.linear_model import LinearRegression



model = LinearRegression()
model.fit(train.drop('price', axis=1), train['price'])
predictions = model.predict(test)
sample_submission.head()
sample_submission['price'] = predictions
sample_submission.to_csv('my_submission.csv', index=False)