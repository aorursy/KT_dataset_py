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
#Importing Libraries



import csv

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

#Import Data File i.e. "HouseSalesPrediction.csv"



df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv") 



#Confirming if loaded correctly in dataframe



df.head(5)

#Checking if there is no Null data



df.isnull().sum()
#Creating Correlation Matrix



correlation_matrix = df.corr() 

  

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(correlation_matrix, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

#Setting up input & output variables



X = df[['price','bedrooms']].values

y = df['sqft_living'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)



#Validating data output



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Import LinearRegression module



from sklearn.linear_model import LinearRegression

model = LinearRegression()

output_model = model.fit(X_train, y_train)



#Check Output Model



output_model



#Predict the test result set



y_predictor = model.predict(X_test)



#Print Prediction



print (y_predictor)

#Import Metrics



from sklearn.metrics import mean_squared_error, r2_score

coef = mean_squared_error(y_test, y_predictor)

r2 = r2_score(y_test, y_predictor)



#Print Output



print ("Mean Squared Error is:", coef)

print ("R Squared is:", r2)

#Generating Graph



sns.scatterplot(y_test , y_predictor)