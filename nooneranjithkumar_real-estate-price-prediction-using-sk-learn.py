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
data = pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')

data.head(5)
data.info()
data.describe()
data.isnull().sum()             #check null values
data.drop(['No'], inplace=True , axis =1)         #by default it takes axis = 0

#inplace attribute places the dataframe into same variable after execution , above line code similar to

#data = data.drop(['No'], axis =1)
#renaming of columns for better user-fiendly

data.rename(columns={

    'X1 transaction date': 'Date',

    'X2 house age': 'Age',

    'X3 distance to the nearest MRT station':'Nearest_Station_Distance',

    'X4 number of convenience stores':'Num_Stores',

    'X5 latitude':'latitude',

    'X6 longitude':'longitude',

    'Y house price of unit area':'Price_Unit_Area',

}, inplace=True)

data.head(5)
X = data.drop(['Price_Unit_Area'], axis = 1)

Y = data['Price_Unit_Area']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)
from sklearn.linear_model import LinearRegression
model = LinearRegression()        #Creating object for LinearRegression class

model.fit(x_train, y_train)
model.score(x_test, y_test)
y_pred = model.predict(x_test)

y_test_pred = pd.DataFrame(y_test)

y_test_pred['Predicted_Price_Unit_Area'] = y_pred

y_test_pred.sample(10)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
data['Predicted_Price_Unit_Area'] = model.predict(X)

data.head(5)