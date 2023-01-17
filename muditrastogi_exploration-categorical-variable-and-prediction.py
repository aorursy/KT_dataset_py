import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/diamonds.csv")

data.head()
data.columns
print(len(data))

data.dtypes
data.describe()
obj_df = data.select_dtypes(include=['object']).copy()

obj_df.head()
temp_df = pd.get_dummies(obj_df, columns=["clarity", "color", "cut"])

encoded_df = temp_df.select_dtypes(include=['uint8']).copy()

encoded_df.head()
color_name = set(data['color'])

cut_name = set(data['cut'])

clarity_name = set(data['clarity'])

print ("Color List", color_name)

print("Cut List", cut_name)

print("Clarity List", clarity_name)
data = data.drop(['clarity', 'color','cut'], axis = 1)

complete_data = pd.concat([data, encoded_df], axis =1)

complete_data.head()
X = complete_data.drop(['price', 'Unnamed: 0'], axis =1)

print(X.head())

Y = complete_data['price']

Y.shape, X.shape

X.info(), Y.dtypes
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.30, random_state=3)

y_train.head()
print ("No. of rows considered as training : %d " % len(x_train))

print ("No. of rows considered as test : %d " % len(x_test))



x_train.shape

y_train.shape
from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(x_train, y_train)