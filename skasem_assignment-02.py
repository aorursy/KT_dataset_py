import numpy as np

import matplotlib.pyplot as ptl

import pandas as pd



# import the dataset

dataset = pd.read_csv('fer2013.csv')



#dependent valudes

X = dataset.iloc[:, 1:3] 



#independent,LABEL

y = dataset.iloc[:, 0]



#taking care of missing data using mod,mean,midean

# from sklearn.preprocessing import Imputer

# imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)





#splitting dataset into training and testing 

from sklearn.model_selection import train_test_split

X_train, X_test, y_traing, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


