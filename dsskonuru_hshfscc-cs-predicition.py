import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # plotting



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 



from sklearn.neural_network import MLPRegressor # neural network regressor

from sklearn.tree import DecisionTreeRegressor # decision tree regressor



import sklearn.metrics as metrics

from scipy import stats



def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



# Input data files are available in the "../input/" directory

path = "../input/HSHFSCC.xlsx"

dataset = pd.read_excel(path)

dataset.info()
X = dataset.ix[:,(1,2,3)]

y = dataset.ix[:,(4)]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)



scaler = StandardScaler()

scaler.fit(X_train) # training data is used for scaling

StandardScaler(copy=True, with_mean=True, with_std=True)



X_train = scaler.transform(X_train) # apply the transformations to the data

X_test = scaler.transform(X_test)
# --- making the regression model

regr = DecisionTreeRegressor(max_depth=5)

regr.fit(X_train, y_train)



# --- predictions using the model

pred = regr.predict(X_test)

print('Original Values: ',y_test.values)

print('Predicted Values: ',pred)



slope, intercept, r_value, p_value, std_err = stats.linregress(pred,y_test.values)

print('RMSE: ',rmse(pred,y_test.values), 'R2 Value: ',r_value**2)
fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Steel% vs Compressive Strength')

ax1.scatter(X_test[:,0], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,0],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()



fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Glass% vs Compressive Strength')

ax1.scatter(X_test[:,1], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,1],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()



fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Setting time vs Compressive Strength')

ax1.scatter(X_test[:,2], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,2],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()
# TRAIN THE MODEL

mlp = MLPRegressor(hidden_layer_sizes=(10,3),max_iter=10000, early_stopping=False, random_state=356)

# trail and error is used to determine the random state



# PREDICTIONS

mlp.fit(X_train,y_train)

pred = mlp.predict(X_test)

print('Original Values: ',y_test.values)

print('Predicted Values: ',pred)



slope, intercept, r_value, p_value, std_err = stats.linregress(pred,y_test.values)

print('RMSE: ',rmse(pred,y_test.values), 'R2 Value: ',r_value**2)

fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Steel% vs Compressive Strength')

ax1.scatter(X_test[:,0], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,0],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()



fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Glass% vs Compressive Strength')

ax1.scatter(X_test[:,1], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,1],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()



fig = plt.figure()

ax1 = fig.add_subplot(111)

plt.title('Setting time vs Compressive Strength')

ax1.scatter(X_test[:,2], pred, c='b', marker="s", label='Original Values')

ax1.scatter(X_test[:,2],y_test, c='r', marker="o", label='NN Prediction')

plt.legend()

plt.show()

    