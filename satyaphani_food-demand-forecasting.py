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
from sklearn.preprocessing import LabelEncoder

import os

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/food-demand-forecasting/train.csv')

print('Training data shape: ', train.shape)

train.head()
train.info()
test = pd.read_csv('/kaggle/input/food-demand-forecasting/test.csv')

print('Testing data shape: ', test.shape)

test.head()
centers = pd.read_csv('/kaggle/input/food-demand-forecasting/fulfilment_center_info.csv')

print('Testing data shape: ', centers.shape)

centers.head()
meal = pd.read_csv('/kaggle/input/food-demand-forecasting/meal_info.csv')

print('Testing data shape: ', meal.shape)

meal.head()
meal.info()
sample_submission = ('/kaggle/input/food-demand-forecasting/sample_submission.csv')
plt.figure(figsize=(20,10))

c=train.corr()

sns.heatmap(c,cmap="BrBG",annot=True)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_log_error



# Create target object and call it y

y = train.num_orders
# Create target object and call it y

y = train.num_orders



# Create X

features = ['id','week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion','homepage_featured']

X = train[features]





# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Specify Model

client_model = DecisionTreeRegressor(random_state=1)

# Fit Model

client_model.fit(train_X, train_y)
features = ['id','week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion','homepage_featured']



test_X = test[features]



# make predictions which we will submit. 



test_preds = client_model.predict(test_X)

print(mean_squared_log_error(test.id, test_preds))


output = pd.DataFrame({'id': test.id,

                       'num_orders': test_preds})



output.to_csv('submission.csv', index=False)  