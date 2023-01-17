import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



#import dataset

dataset = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')



Y = dataset.iloc[:,7].values

X= dataset.drop(['Total_Protiens','Gender'], axis=1).iloc[:,:].values

#taking care of missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean', verbose=0)

imputer= imputer.fit(X[:,:])

X[:,:]= imputer.transform(X[:,:])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8)



#fitting Regression Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#predicting test set results

y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
