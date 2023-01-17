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
# save filepath to variable for easier access

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data

melbourne_data = pd.read_csv(melbourne_file_path) 

# print a summary of the data in Melbourne data

melbourne_data.describe()
melbourne_data.columns
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)

# We'll learn to handle missing values in a later tutorial.  

# So we will take the simplest option for now, and drop houses from our data.

# dropna drops missing values (think of na as "not available")

melbourne_data = melbourne_data.dropna(axis=0)
y=melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

#print(melbourne_features)
X = melbourne_data[melbourne_features]

#print(X)



X.describe()

X.head()
#import the DecisionTreeregressor from the package sklearn.tree

from sklearn.tree import DecisionTreeRegressor



#specify the model

#For model reproducibility, set a numeric value for random_state when specifying the model

melbourne_model = DecisionTreeRegressor(random_state=1)



#Fit the model

melbourne_model.fit(X,y)

predictions = melbourne_model.predict(X)



#checking the predictions with the Price that I stored in a variable y

print(predictions)

print(y)


