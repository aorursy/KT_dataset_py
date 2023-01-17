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
#I loaded my data from home data for ML course

home_data_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(home_data_path)
home_data.head()
home_data.describe()
#checking the columns 

home_data.columns
#import

from sklearn.model_selection import train_test_split

#creating the target data which is called y

y = home_data.SalePrice

#creating features

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



#split the data into train and val

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)







#import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

home_model = RandomForestRegressor(random_state = 1)
home_model.fit(train_X,train_y)
home_prediction = home_model.predict(val_X)
print(home_prediction)
#importing metrics from sklearn

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(home_prediction,val_y)
print(val_mae)
y.head()