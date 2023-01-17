import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Loading Data
df = pd.read_csv('../input/housingdata.csv', header = None)
# Data disposition
df.head()
# Slicing, predictors and predict variable
X = df.drop(13, axis = 1)
y = df[13]
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Scaling Data for ANN
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Number of features to input
num_features = len(X_train[1,:])
# ANN with Keras
np.random.seed(10)
classifier = Sequential()
     # better values with tanh against relu, sigmoid...
classifier.add(Dense(13, kernel_initializer = 'uniform', activation = 'tanh', input_dim = num_features)) 
classifier.add(Dense(1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error')        # metrics=['mse','mae']
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=500)  # ignored
history_mse = classifier.fit(X_train, y_train, epochs = 100, callbacks = [early_stopping_monitor], verbose = 0, validation_split = 0.2)

print('Loss:    ', history_mse.history['loss'][-1], '\nVal_loss: ', history_mse.history['val_loss'][-1])
# EVALUATE MODEL IN THE TEST SET
score_mse_test = classifier.evaluate(X_test, y_test)
print('Test Score:', score_mse_test)

# EVALUATE MODEL IN THE TRAIN SET
score_mse_train = classifier.evaluate(X_train, y_train)
print('Train Score:', score_mse_train)
plt.figure(figsize=(15, 6))
plt.plot(history_mse.history['loss'], lw =3, ls = '--', label = 'Loss')
plt.plot(history_mse.history['val_loss'], lw =2, ls = '-', label = 'Val Loss')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('MSE')
plt.legend()
#Converting the first line of the dataset
linha1 = np.array([0.00632 ,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296,15.3,396.90,4.98]).reshape(1,-1)
# Scaling the first line to the same pattern used in the model
linha1 = sc_X.transform(linha1)
# Predicted value by model
y_pred_mse_1 = classifier.predict(linha1)
print('Predicted value: ',y_pred_mse_1)
print('Real value: ','24.0')
# Predicting the 15 value at test set
newValues = np.array([0.62739, 0.0,8.14,0,0.538,5.834,56.5,4.4986,4,307,21.0,395.62,8.47]).reshape(1, -1)
# Scaling new values to the same pattern used in the model
newValues = sc_X.transform(newValues)
# Predictied value by model
y_pred_mse_15 = classifier.predict(newValues)
print('Predicted value: ',y_pred_mse_15)
print('Real value: ','19.9')
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
# ANN with Keras
np.random.seed(10)
classifier = Sequential()
     # better values with tanh agains relu, sigmoid...
classifier.add(Dense(13, kernel_initializer = 'uniform', activation = 'tanh', input_dim = num_features)) 
classifier.add(Dense(1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'sgd', loss = root_mean_squared_error)        # metrics=['mse','mae']
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=500)  # ignored
history = classifier.fit(X_train, y_train, epochs = 100, callbacks = [early_stopping_monitor], verbose = 0, validation_split = 0.2)

print('Loss:    ', history.history['loss'][-1], '\nVal_loss: ', history.history['val_loss'][-1])
# EVALUATE MODEL IN THE TEST SET
score_rmse_test = classifier.evaluate(X_test, y_test)
print('Test Score:', score_rmse_test)

# EVALUATE MODEL IN THE TRAIN SET
score_rmse_train = classifier.evaluate(X_train, y_train)
print('Train Score:', score_rmse_train)
plt.figure(figsize=(15, 6))
plt.plot(history.history['loss'], lw =3, ls = '--', label = 'Loss')
plt.plot(history.history['val_loss'], lw =2, ls = '-', label = 'Val Loss')
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.title('RMSE')
plt.legend()
#Converting the first line of the dataset
linha1 = np.array([0.00632 ,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296,15.3,396.90,4.98]).reshape(1,-1)
# Scaling the first line to the same pattern used in the model
linha1 = sc_X.transform(linha1)
# Predicted value by model
y_pred_rmse_1 = classifier.predict(linha1)
print('Predicted value: ',y_pred_rmse_1)
print('Real value: ','24.0')
# Predicting the 15 value at test set
newValues = np.array([0.62739, 0.0,8.14,0,0.538,5.834,56.5,4.4986,4,307,21.0,395.62,8.47]).reshape(1, -1)
# Scaling new values to the same pattern used in the model
newValues = sc_X.transform(newValues)
# Predictied value by model
y_pre_rmse_15 = classifier.predict(newValues)
print('Predicted value: ',y_pre_rmse_15)
print('Real value: ','19.9')
models = pd.DataFrame({
    'Model': ['Test  Set Score', 'Train Set Score', 'Predict first Line [24.0]', 'Predict Test Set Value [19.9]',
              'Last Epoch Loss', 'Last Epoch Val Loss'],
    'MSE': [np.sqrt(score_mse_test), np.sqrt(score_mse_train), y_pred_mse_1[0], y_pred_mse_15[0],
            np.sqrt(history_mse.history['loss'][-1]), np.sqrt(history_mse.history['val_loss'][-1])],
    'RMSE': [score_rmse_test, score_rmse_train, y_pred_rmse_1[0], y_pre_rmse_15[0], history.history['loss'][-1],
             history.history['val_loss'][-1]]
})
models