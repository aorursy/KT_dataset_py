# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/deep-learning-az-ann/Churn_Modelling.csv")

data.head()
data.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace = True)
data.head(2)
# CreditScore, Age, Tenure, Balance, EstimatedSalary to be rescaled:



for each in ["CreditScore", "Age","Tenure", "Balance", "EstimatedSalary"]:

    data[each] = (data[each] - np.min(data[each])) / (np.max(data[each])-np.min(data[each]))

data.head()
# looking at current types:

data.info()
# converting type of some features to category

for each in ["Geography","Gender","NumOfProducts","HasCrCard","IsActiveMember","Exited"]:

    data[each] = data[each].astype("category")
# types after conversion

data.info()
data.info()
data = pd.get_dummies(data, columns = ["Geography","Gender", "NumOfProducts"])
data.head()
data.info()
X_train = data.drop(columns = ["Exited"], axis=1)
y_train = data["Exited"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X_train, 

    y_train,

    test_size = 0.33,

    random_state = 42

)



print("Length of X_train: ",len(X_train))

print("Length of X_test: ",len(X_test))

print("Length of y_train: ",len(y_train))

print("Length of y_test: ",len(y_test))
print(

    "Shape of X_train: ",np.shape(X_train),

    "\nShape of y_train: ",np.shape(y_train)

)
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X_train.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X_test.columns.values)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential     # Neural network library

from keras.layers import Dense          # layer library

def create_model():

    

    # create model

    model = Sequential()

    

    # adding input layer

    model.add(Dense(units = 12, kernel_initializer = "uniform", activation = "relu", input_dim = 16))

    

    # adding layer

    model.add(Dense(units = 8, kernel_initializer = "uniform", activation = "relu"))

    

    # adding layer

    model.add(Dense(units = 4, kernel_initializer = "uniform", activation = "relu"))

        

    # adding output layer

    model.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

    

    # compile model

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    

    return model    
model1 = KerasClassifier(build_fn = create_model, epochs=15)
history1 = model1.fit(X_train,y_train)
model2 = KerasClassifier(build_fn = create_model, epochs=15, batch_size = 10)

history2 = model2.fit(X_train, y_train)
plt.subplots(figsize = (10,6))

plt.plot(history1.history["accuracy"], label = "Batch size = 32")

plt.plot(history2.history["accuracy"], label = "Batch size = 10")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracies")

plt.title("Affects of batch size on accuracy on ANN")

plt.grid(axis = "both")



plt.legend()

plt.show()

model = KerasClassifier(build_fn = create_model, epochs=15, batch_size = 10)

kfold = StratifiedKFold(n_splits = 15, shuffle = True, random_state = 42)

accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = kfold)
plt.subplots(figsize = (10,6))

plt.plot(accuracies)

plt.xlabel("K-fold values of Cross Validation Score")

plt.ylabel("Accuracies")

plt.title("Cross Validation Accuracies vs K-Folds of ANN")

plt.grid(axis = "both")



plt.show()
print("Best accuracy : {} @ k-fold value of {}".format(round(accuracies.max()*100,2),accuracies.argmax()))
from sklearn.model_selection import GridSearchCV



def create_model1(optimizer="rmsprop", init="glorot_uniform"):

        

    # create model

    model = Sequential()

    

    # adding input layer

    model.add(Dense(units = 12, kernel_initializer = init, activation = "relu", input_dim = 16))

    

    # adding layer

    model.add(Dense(units = 8, kernel_initializer = init, activation = "relu"))

    

    # adding layer

    model.add(Dense(units = 4, kernel_initializer = init, activation = "relu"))

        

    # adding output layer

    model.add(Dense(units = 1, kernel_initializer = init, activation = "sigmoid"))

    

    # compile model

    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

    

    return model  

    
# create model

model_new = KerasClassifier(build_fn = create_model1, epochs = 15, batch_size = 32)



# grid search epochs, batch size and optimizer

optimizers = ['rmsprop', 'adam']

init = ['glorot_uniform', 'uniform']



param_grid = dict(optimizer = optimizers, init = init)

grid = GridSearchCV(estimator = model_new, param_grid = param_grid)



result = grid.fit(X_train, y_train)

# summarize results

print("Best: %f using %s" % (result.best_score_, result.best_params_))

means = result.cv_results_['mean_test_score']

stds = result.cv_results_['std_test_score']

params = result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))