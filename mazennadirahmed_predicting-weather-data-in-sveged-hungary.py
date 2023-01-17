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



# Read data into weatherData variable

weatherData = pd.read_csv("../input/weatherhistory/weatherHistory.csv") 



# Display the first few rows of our Data

weatherData.head()
weatherData.describe() # Gives basic statistics about our data
# Import preprocessing functions

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn import preprocessing





# Define Model's features

weatherFeatures = ["Temperature (C)","Apparent Temperature (C)","Wind Speed (km/h)", 

                   "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)"]



X = weatherData[weatherFeatures]

y = weatherData.Humidity



# Scale data to fit standard normal distribution for accurate application of linear regression with regularization

X_scaled  = preprocessing.scale(X) 



# Polynomial Features enable us to capture non-linear relationships

# Currently set to 1 - CHANGE TO 7 after reading evaluation section below

poly = PolynomialFeatures(1) 



X_final = poly.fit_transform(X_scaled)





# Split data into training and test data

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.10, random_state=42) 
from sklearn import linear_model



# Implement Ridge Regress - Linear regression + l2 regularization

# Regularization parameter set to 0.5

regr = linear_model.Ridge(alpha = 0.5) 





# Fit Model to Data

regr.fit(X_train, y_train)



# Produce a set of predicted Humidity values from our test data

y_pred = regr.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score



# Display Model Intercept

print("Intercept: ", regr.intercept_)



# Display Model coefficients

print('Coefficients: \n', len(regr.coef_ ))





print('Mean squared error: %.3f' % mean_squared_error(y_test,  y_pred))





# r2_score (Coefficient of determination) is a great evaluation metric - read more below

print('Coefficient of determination: %.3f' % r2_score(y_test, y_pred))
# Record a new set of weather observations, ensure oder of entry matches order of weatherFeatures defined above

weatherObs = [[32, 31.4, 44, 344, 13, 1020.33]]



# Apply the same preprocessing steps

weatherObs_scaled = preprocessing.scale(weatherObs) 

weatherObs_final = poly.fit_transform(weatherObs_scaled)



# Use our model to make a prediction for Humidity

y_pred = regr.predict(weatherObs_final)

y_pred