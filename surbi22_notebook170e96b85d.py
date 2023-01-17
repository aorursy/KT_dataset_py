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
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
home_data = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(home_data)

# Create target object and call it y

y = home_data.SalePrice



# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
X.describe()
model = DecisionTreeRegressor(random_state=1)

model.fit(X, y)
print("The predictions are")

print(model.predict(train_X))
val_y.shape
y_pred = model.predict(val_X)
y_pred.shape
from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(val_y ,y_pred)

error
score = accuracy_score(val_y,y_pred)

score