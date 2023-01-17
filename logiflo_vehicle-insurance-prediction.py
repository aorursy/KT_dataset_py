# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_test = '/kaggle/input/health-insurance-cross-sell-prediction/test.csv'
path_train = '/kaggle/input/health-insurance-cross-sell-prediction/train.csv'
data_test = pd.read_csv(path_test)
data_train = pd.read_csv(path_train)
data_test.head()
data_test.info()
print('-'* 50)
data_train.info()
print(data_test.Gender.unique())
print(data_test.Vehicle_Age.unique())
print(data_test.Vehicle_Damage.unique())
cleanup_nums = {'Gender': {'Male': 0, 'Female': 1},
                'Vehicle_Age': {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2},
                'Vehicle_Damage': {'No': 0, 'Yes': 1}}

data_test.replace(cleanup_nums, inplace=True)
data_train.replace(cleanup_nums, inplace=True)
print(data_test.Gender.unique())
print(data_test.Vehicle_Age.unique())
print(data_test.Vehicle_Damage.unique())
sns.countplot(x = "Gender", hue='Response', data = data_train)

sns.countplot(x = "Vehicle_Damage", hue='Response', data = data_train)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
y = data_train.Response
# Create X
features = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Vintage']
X = data_train[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)
test_X = data_test[features]

test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'id': data_test.id,
                       'Response': test_preds})
output.to_csv('submission.csv', index=False)