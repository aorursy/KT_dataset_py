import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

from tensorflow import keras
cust = pd.read_csv('../input/churn-for-bank-customers/churn.csv')
# print the first five rows of the data set

cust.head()
# print the shape of the dataset

cust.shape
# few detail about data

cust.info()
# statistical detail about data

cust.describe()
# check for missing values

cust.isna().sum()
# we wont take the first two columns (row id and customer id)

x = cust.iloc[:, 3:13]

# target variable
y = cust.iloc[:, -1]
x.Gender = pd.factorize(x.Gender)[0]

x.Geography = pd.factorize(x.Geography)[0]
x.head()
# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
display(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, PReLU, ELU, Dropout
# Initialising the ANN

classify = Sequential()
# Adding the input layer and the first hidden layer

classify.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation ='relu', input_dim =10))

# Adding the second hidden layer

classify.add(Dense(units = 6, kernel_initializer ='he_uniform' ,activation ='relu'))

# Adding the output layer

classify.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
classify.summary()
# Compiling the ANN

classify.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='accuracy')
# Fitting the ANN to the Training set

model = classify.fit(x_train, y_train, validation_split=0.20, batch_size=10, epochs=100)
# list all data in histroy

print(model.history.keys())
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

y_predict = classify.predict(x_test)
y_predict = (y_predict > 0.5)


print(confusion_matrix(y_test, y_predict))

print('\nAccuracy: ', accuracy_score(y_predict,y_test))
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
lg = LogisticRegressionCV(cv=3)

lg.fit(x_train, y_train)

p_predict_train = lg.predict(x_train)

p_predict_test = lg.predict(x_test)

print(accuracy_score(p_predict_train, y_train))
print(accuracy_score(p_predict_test, y_test))
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import LeakyReLU, Activation, Dropout, Dense, Embedding, Flatten, BatchNormalization
from keras.activations import relu, sigmoid
def create_model(layers, activation):
    model = Sequential()
    
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)
layers = [(20,), (30, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']

param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[100])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(x_train, y_train)
print(grid_result.best_score_,grid_result.best_params_)
