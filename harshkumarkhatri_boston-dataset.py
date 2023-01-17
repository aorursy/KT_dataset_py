from sklearn import linear_model
import numpy as np
print("Hello")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataset_train=pd.read_csv("boston_train.csv")
dataset_test=pd.read_csv("boston_test.csv")
dataset_predict=pd.read_csv("boston_predict.csv")
np.shape(dataset_train)
np.shape(dataset_predict)
np.shape(dataset_test)


pd.DataFrame(dataset_train.isnull().sum())
# Checking for null value sin testdataset
pd.DataFrame(dataset_test.isnull().sum())

# Checking for null values in predict datsset

pd.DataFrame(dataset_predict.isnull().sum())

dataset_train=dataset_train.dropna()
dataset_train.isnull().sum()
# dropping na values from test dataser
dataset_test=dataset_test.dropna()
dataset_test.isnull().sum()
x_train=dataset_train.drop('TAX',axis=1)
# print(dataset_train_x.head().values)
y_train=dataset_train['TAX']
y_train.head()
# initializing the model
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
# y_predict=reg.predict()
dataset=pd.read_csv("HousingData.csv")
dataset.shape
from sklearn.model_selection import train_test_split
dataset.isnull().sum()
dataset=dataset.dropna()
dataset.isnull().sum()
x=dataset.drop('TAX',axis=1)
y=dataset['TAX']
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
x_train.head
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train,sample_weight=None)

y_pred=reg.predict(x_test)
reg.score(x_test,y_test)
# reg.score(x_test,y_pred)