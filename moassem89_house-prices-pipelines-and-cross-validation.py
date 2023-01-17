# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

home_data = pd.read_csv("../input/train.csv")

home_test = pd.read_csv("../input/test.csv")



y = home_data.SalePrice

X = home_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])



from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer



my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())



my_pipeline.fit(train_X, train_y)

predictions = my_pipeline.predict(test_X)
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))