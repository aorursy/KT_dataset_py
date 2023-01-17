!pip install GML
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
from GML.Ghalat_Machine_Learning import Ghalat_Machine_Learning
# Read the data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



train_X = train[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]

train_Y = train.SalePrice



# Read the test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']]



# # Use the model to make predictions

# predicted_prices = my_model.predict(test_X)

# # We will look at the predicted prices to ensure we have something sensible.

# print(predicted_prices)
gml = Ghalat_Machine_Learning()


new_X,y = gml.Auto_Feature_Engineering(train_X,train_Y,type_of_task='Regression',test_data=None,

                                                          splits=6,fill_na_='median',ratio_drop=0.2,

                                                          generate_features=True,feateng_steps=2)
new_X
from sklearn.neural_network import MLPRegressor
best_model = gml.GMLRegressor(new_X,y,neural_net='Yes',epochs=100,models=[MLPRegressor()],verbose=False)