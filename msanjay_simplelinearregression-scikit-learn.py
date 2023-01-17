import pandas as pd
import numpy as np
#Read training and test data
df_train = pd.read_csv('../input/train.csv')
df_test= pd.read_csv('../input/test.csv')
#Check for null values
print(df_train.isnull().sum())
print(df_test.isnull().sum())
#drop if at all any missing values in training
df_train.dropna(inplace=True)
x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']
#convert the pandas series objects into ndarrays.
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
#reshape x_train and x_test to feed into sklearn.
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#fit the model  on training data.
clf = LinearRegression()
clf.fit(x_train,y_train)
#run the model on test and print the r-squared value
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
