# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.ensemble import RandomForestRegressor



# Import the data

train = pd.read_csv('../input/train.csv')

train_y = train.SalePrice



predictors_columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

train_X = train[predictors_columns]



model = RandomForestRegressor()

model.fit(train_X, train_y)



# read test data

test = pd.read_csv('../input/test.csv')

test_X = test[predictors_columns]



predicted_prices = model.predict(test_X)

print(predicted_prices)



my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice':predicted_prices})

my_submission.to_csv('random_forest.csv', index=False)


