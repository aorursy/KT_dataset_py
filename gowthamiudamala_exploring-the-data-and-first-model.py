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
import pandas as pd
data=pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

data.dropna(axis=0)
data.head()
data.describe()
data.columns
from sklearn.model_selection import train_test_split

train_data,test_data=train_test_split(data,random_state=0,train_size=0.8,test_size=0.2)
features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

train_y=train_data['Price']

train_data[features].describe()
from sklearn.tree import DecisionTreeRegressor

# defining my model

my_model=DecisionTreeRegressor()

#fitting my model

my_model.fit(train_data[features],train_y)

#making predictions on my model

test_y=pd.Series(my_model.predict(test_data[features]))

type(test_y)
print(train_data[features].head())

print("predicted values are")

print(my_model.predict(train_data[features].head()))
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(test_data['Price'],test_y))