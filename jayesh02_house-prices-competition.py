import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# trainining data



import pandas as pd



base_dir = "/kaggle/input/home-data-for-ml-course/"



input_data = pd.read_csv(base_dir + "train.csv")

test_data = pd.read_csv(base_dir + "test.csv")



# Data Description

data_description = open(base_dir + "data_description.txt", "r")

#print(data_description.read())



# Column list of training data

#print(input_data.columns)

# Common sense selection

features = ['MSZoning', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

             'HouseStyle', 'OverallQual', 'OverallCond', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'SaleType', 'SaleCondition']

#Spit input data to training_data and test_data



y = input_data['SalePrice']



# Convert text columns to multiple columns, example:

# | Color |

#   Red

#   Green

#   Blue

# after the converstion it will be

#| Color Red | Color Blue | Color Green |

#      1             0           0

#      0             1           0

#      0             0           1

training_data = input_data[features]

output_data = test_data[features]

input_length = len(output_data)

concat = pd.concat(objs=[output_data, training_data], axis=0)

concat = pd.get_dummies(concat)

output_data = concat[:input_length].copy()

training_data = concat[input_length:].copy()



X = training_data





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
from sklearn.tree import DecisionTreeRegressor



dtr_model_ = DecisionTreeRegressor(random_state=1)

dtr_model.fit(X_train, y_train)
# Mean absolute error

from sklearn.metrics import mean_absolute_error



mea = mean_absolute_error(y_test, dtr_model.predict(X_test))

print(mea)
#RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor



rfr_model = RandomForestRegressor(random_state=10)

rfr_model.fit(X_train, y_train)



mea = mean_absolute_error(y_test, rfr_model.predict(X_test))

print(mea)
# Final training on entire data set

rfr_model.fit(X, y)



preds = rfr_model.predict(output_data)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)