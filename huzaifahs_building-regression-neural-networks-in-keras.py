import pandas as pd #pandas for dataframe
import numpy as np #numpy for arrays
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
df.head()
dfcolumns = df.columns #lets extract the columns

predictors = df[dfcolumns[dfcolumns != 'Strength']] # all columns except Strength
target = df['Strength'] # Strength column
predictors.head()
target.head()
ncols = predictors.shape[1]
import keras
from keras.models import Sequential
from keras.layers import Dense
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(ncols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = regression_model() #declaring the model
errorlist=[] #list for keeping 50 mean squared error values

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for x in range(50):
    print ("Iteration ", x)
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0) ##returns the value for the mean squared error
    errorlist.append(score)
    x = x+1
print (errorlist)
print ("---------------------------------")
print ("The mean of MSEs is: ", np.mean(errorlist))
print ("The Std. Deviation of MSEs is: ", np.std(errorlist))
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
errorlistB=[]

for x in range(50):
    print ("Iteration ", x)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0) ##returns the value for the mean squared error
    errorlistB.append(score)
    x = x+1
print (errorlistB)
print ("---------------------------------")
print ("The mean of MSEs is: ", np.mean(errorlistB))
print ("The Std. Deviation of MSEs is: ", np.std(errorlistB))
print ("The Mean of MSEs of part A: ", np.mean(errorlist))
print ("The Mean of MSEs of part B: ", np.mean(errorlistB))
print ("Difference: ", (np.mean(errorlist) - np.mean(errorlistB)))
errorlistC=[]

for x in range(50):
    print ("Iteration ", x)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0) ##returns the value for the mean squared error
    errorlistC.append(score)
    x = x+1
print (errorlistC)
print ("---------------------------------")
print ("The mean of MSEs is: ", np.mean(errorlistC))
print ("The Std. Deviation of MSEs is: ", np.std(errorlistC))
print ("The Mean of MSEs of part B: ", np.mean(errorlistB))
print ("The Mean of MSEs of part C: ", np.mean(errorlistC))
print ("Difference: ", (np.mean(errorlistB) - np.mean(errorlistC)))
# define regression model
def regression_modelD():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(ncols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = regression_modelD()
errorlistD=[]

for x in range(50):
    print ("Iteration ", x)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0) ##returns the value for the mean squared error
    errorlistD.append(score)
    x = x+1
print (errorlistC)
print ("---------------------------------")
print ("The mean of MSEs is: ", np.mean(errorlistD))
print ("The Std. Deviation of MSEs is: ", np.std(errorlistD))
print ("The Mean of MSEs of part B: ", np.mean(errorlistB))
print ("The Mean of MSEs of part C: ", np.mean(errorlistD))
print ("Difference: ", (np.mean(errorlistB) - np.mean(errorlistD)))