# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
train_data_path = '../input/titanic/train.csv'

training_data = pd.read_csv(train_data_path)
training_data.head()
training_data.columns
training_data.describe()
training_data.isnull().sum()
training_data.isnull().sum().sum()
training_data.tail()
training_data1 = training_data.dropna()

training_data1
training_data1.isnull().sum().sum()
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
# Create target object and call it y

y = training_data1.Survived
# Create X

features = ['PassengerId', 'Pclass', 'Age']

x = training_data1[features]

# Split into validation and training data

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)



# Specify Model

training_model = DecisionTreeRegressor(random_state=1)

# Fit Model

training_model.fit(train_x, train_y)
print("Making predictions for the following 5 Passengers:")

print(x.head())

print("The predictions are")

print(training_model.predict(x.head()))
print("The predictions are")

print(training_model.predict(x.head()))

print(y.head())
from sklearn.metrics import mean_absolute_error

predicted_survival_rates = training_model.predict(x)

mean_absolute_error(y, predicted_survival_rates)





val_predictions = training_model.predict(val_x)

print(mean_absolute_error(val_y, val_predictions))

print('Top five survival predictions', val_predictions[0:5])

print('top five validated actual survival', val_y[0:5])
training_model_on_full_data = RandomForestRegressor()

training_model_on_full_data.fit(x, y)
testing_data_path = '../input/titanic/test.csv'

test_data = pd.read_csv(testing_data_path)
test_data.head()
test_data1 = test_data.dropna()

test_data1.head()
test_x = test_data1[features]

test_preds = training_model_on_full_data.predict(test_x)
output = pd.DataFrame({'Id': test_data1.PassengerId,

                       'Survival': test_preds})

output.to_csv('submission.csv', index=False)

output.head()