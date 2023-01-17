import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read the csv file 
admission_df=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
admission_df.head()
# Let's drop the serial no.
admission_df.drop('Serial No.',axis=1 , inplace=True)
admission_df
admission_df

# checking the null values
admission_df.isnull().sum()
# Check the dataframe information
admission_df.info()
# Statistical summary of the dataframe
admission_df.describe()
# Grouping by University ranking 
df_university=admission_df.groupby(by='University Rating').mean()
df_university
admission_df.hist(bins=30 , figsize=(20,20))
sns.pairplot(admission_df)

corr_matrix =admission_df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix , annot=True)
plt.show()
admission_df.columns
X = admission_df.drop(columns = ['Chance of Admit '])
y = admission_df['Chance of Admit ']
X.shape
y.shape
y
X = np.array(X)
y = np.array(y)
y= y.reshape(-1,1)
y.shape
# scaling the data before training the model
from sklearn.preprocessing import StandardScaler , MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size= 0.15)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X_train, y_train)
accuracy_LinearRegression = LinearRegression_model.score(X_test, y_test)
accuracy_LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 7))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()
ANN_model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size = 20, validation_split = 0.2)
result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result
print("Accuracy : {}".format(accuracy_ANN))
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
# Decision tree builds regression or classification models in the form of a tree structure. 
# Decision tree breaks down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. 
# The final result is a tree with decision nodes and leaf nodes.
# Great resource: https://www.saedsayad.com/decision_tree_reg.htm

from sklearn.tree import DecisionTreeRegressor
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)
accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
accuracy_DecisionTree
from sklearn.ensemble import RandomForestRegressor
RandomForest_model = RandomForestRegressor(n_estimators=100, max_depth = 10)
RandomForest_model.fit(X_train, y_train)
accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
accuracy_RandomForest
y_predict = LinearRegression_model.predict(X_test)
plt.plot(y_test, y_predict, '^', color = 'r')
y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)
plt.plot(y_test_orig, y_predict_orig, '^', color= 'r')
k = X_test.shape[1]
n = len(X_test)
n
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

