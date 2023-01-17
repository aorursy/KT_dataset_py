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
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'



# Fill in the line below to read the file into a variable home_data

home_data = pd.read_csv(iowa_file_path ) 



#View the dataframe

home_data.describe()
#View all the columns of the datframe

home_data.columns
#Set Target Value

y = home_data.Price

#Set Features value

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = home_data[melbourne_features]

X.describe()





#Returns top 5 rows of data frame

X.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model

melbourne_model = DecisionTreeRegressor()

# Fit model

melbourne_model.fit(train_X, train_y)

#Prediction using test data

print("Making predictions for the following 5 houses:")

print(val_X.head())

print("The predictions are")

print(melbourne_model.predict(val_X.head()))
#Prediction of a single data sample

data = {'Rooms':[4], 'Bathroom':[1],'Landsize':[200],'Lattitude':[-37.7994],'Longtitude':[144.2222]} 



new_input_df = pd.DataFrame(data) 

#Showing data frame of the new input

new_input_df

print("The predictions are")

print(melbourne_model.predict(new_input_df))



# get predicted prices on validation data

val_predictions = melbourne_model.predict(val_X)

#Calculation of Mean Abosulte Error

print(mean_absolute_error(val_y, val_predictions))