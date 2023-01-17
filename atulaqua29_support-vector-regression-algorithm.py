# importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory

# print all the file/directories present at the path

import os

print(os.listdir("../input/"))
# importing the dataset

dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset.head()
dataset.info()
dataset.isnull().sum()
plt.plot(dataset.iloc[:,1:-1],dataset.iloc[:,-1],color='red')

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.title('Position Level VS Salary')

plt.show()
# matrix of features as X and dep variable as Y (convert dataframe to numpy array)

X = dataset.iloc[:,1:-1].values          #Level

Y = dataset.iloc[:,-1].values           #Salary
X.shape

Y.shape
# X.shape = (10,1)

# Y.shape = (10,)



# Feature scaling



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_Y = StandardScaler()

X = sc_X.fit_transform(X)

Y = sc_Y.fit_transform(np.reshape(Y,(len(Y),1))) 

# fit_transform() accepts numpy array of shape [n_samples,n_features] as input
# SVR Regressor



from sklearn.svm import SVR

reg = SVR(kernel='rbf')

reg.fit(X,Y)
# Testing the prediction e.g. on X value = 6.5



y_pred_featured_scaled = reg.predict(sc_X.transform(np.array([[6.5]])))

# reg.predict input type ---> X : {array-like, sparse matrix}, shape (n_samples, n_features)

# Hence, created np array of int 6.5 with shape = (1,1) and then applied the feature scaling



# The precited output would be also scaled value; apply the inverse transformation



y_pred = sc_Y.inverse_transform(y_pred_featured_scaled)

y_pred
# X Axis ---> featured values of X array

# Y Axis ---> featured predicted values for each element in X



plt.plot(X,reg.predict(X),color='blue')

plt.title('Position Level VS Salary')

plt.show()