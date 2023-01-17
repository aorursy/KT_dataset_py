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
import numpy as np

import pandas as pd

import os 

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

data.head(5)
#Checking the Null Values

missing_values = data.isnull()

missing_values.tail
#Data Seems on scatterplot

sns.scatterplot(x='sqft_lot15' , y='sqft_living', data = data.tail(24000))

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
# Train data 

X = data[['sqft_living15','price','sqft_above']]

y = data['sqft_living']
from sklearn.model_selection import train_test_split

#Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=200)
# Import model for fitting

from sklearn.linear_model import LinearRegression

model = LinearRegression()

output_model=model.fit(X_train, y_train)

output_model
#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)