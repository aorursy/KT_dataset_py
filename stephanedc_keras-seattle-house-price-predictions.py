# Importing the libraries
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# Importation de Keras
from keras.models import Sequential   
from keras.layers import Dense        
from keras.layers import Dropout      

# Classifiers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# Configuration
EPOCHS = 100
RATIO_TRAIN_TEST = 0.20
BATCH_SIZE = 16
CROSS_VALIDATION = 3
def compute_score(y_pred, y_test): 
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    var = explained_variance_score(y_test, y_pred)
    r2sc = r2_score(y_test, y_pred)
    
    print("Mean Squarred Error: %.4f" % mse)
    print("Root Mean Squarred Error: %.4f" % rmse)
    print("Mean Absolute Error: %.4f" % mae)
    print("Variance Score (Best possible score is 1): %.4f" % var)
    print("R2Score (Best possible score is 1): %.4f" % r2sc)
def plot_history_detail(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train mean_absolute_error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='mean_squared_error')
    plt.legend()
    plt.ylim([0,1])
# Importing the dataset
dataset = pd.read_csv("../input/home_data(3).csv", delimiter=",")
dataset = dataset.drop(columns=['id', 'date'])
X = dataset.drop(columns=['price']).astype("float64")
y = dataset.price.astype("float64").values.reshape(-1, 1)
print("X Shape is {}".format(X.shape))
# Scaling data
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = pd.DataFrame.from_records(data=X_scaler.fit_transform(X), columns=X.columns)
y_scaled = pd.DataFrame.from_records(data=y_scaler.fit_transform(y))
# Spliting train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = RATIO_TRAIN_TEST, random_state = 0)
print("X_train shape: {}".format(X_train.shape))
# Construct Regressor NN
def build_regressor_NN():
    regressor = Sequential()
    regressor.add(Dense(units=18, activation="relu", kernel_initializer="normal", input_dim=18))
    regressor.add(Dropout(rate=0.1))   
    regressor.add(Dense(units=64, activation="relu", kernel_initializer="normal"))
    regressor.add(Dropout(rate=0.1))
    regressor.add(Dense(units=1, kernel_initializer="normal", activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    regressor.summary()
    return regressor 
# First Training the regressor without Cross Validation
regressor = build_regressor_NN()
history = regressor.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
y_pred = y_scaler.inverse_transform(regressor.predict(X_test))
y_true = y_scaler.inverse_transform(y_test)
plot_history_detail(history)
compute_score(y_pred, y_true)