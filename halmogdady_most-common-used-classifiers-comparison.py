import pandas as pd

import os

data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")

print(data.shape)

data.head()
import seaborn as sns



sns.pairplot(data, hue='target_class')

sns.heatmap(data.corr(), annot=True)
y = data.target_class.values

x_data = data.drop(['target_class'], axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.15, random_state=42)



from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(x_train)  

X_train = scaler.transform(x_train)  

X_test = scaler.transform(x_test)

print(X_train)
from sklearn import linear_model #import the model library

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 500,solver='lbfgs') # sitting model parameters

print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.neighbors import KNeighborsClassifier #import the model library

neigh = KNeighborsClassifier(n_neighbors=3) # sitting model parameters

print("test accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(neigh.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn import tree #import the model library

dt = tree.DecisionTreeClassifier() # sitting model

print("test accuracy: {} ".format(dt.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(dt.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn import svm #import the model library

svm = svm.SVC(gamma='scale') # sitting model parameters

print("test accuracy: {} ".format(svm.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(svm.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.neural_network import MLPClassifier #import the model library

snn = MLPClassifier(solver='lbfgs', alpha=1e-2,hidden_layer_sizes=(8, 8), random_state=1) # sitting model parameters

print("test accuracy: {} ".format(snn.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(snn.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.naive_bayes import GaussianNB #import the model library

gnb = GaussianNB() # sitting model

print("test accuracy: {} ".format(gnb.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(gnb.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.ensemble import BaggingClassifier #import the model library

from sklearn.neighbors import KNeighborsClassifier #import the model library

bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5) # sitting model parameters

print("test accuracy: {} ".format(bagging.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(bagging.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.ensemble import RandomForestClassifier #import the model library

rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0) # sitting model parameters

print("test accuracy: {} ".format(rf.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(rf.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.ensemble import AdaBoostClassifier #import the model library

adab = AdaBoostClassifier(n_estimators=100) # sitting model parameters

print("test accuracy: {} ".format(adab.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(adab.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.ensemble import GradientBoostingClassifier #import the model library

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0) # sitting model parameters

print("test accuracy: {} ".format(gbc.fit(x_train, y_train).score(x_test, y_test))) # printing the results of fitting the model over the testing set

print("train accuracy: {} ".format(gbc.fit(x_train, y_train).score(x_train, y_train))) # printing the results of fitting the model over the training set
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier



clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1,max_iter= 500)

clf2 = RandomForestClassifier(n_estimators=50, random_state=1)

clf3 = GaussianNB()



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):

    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from vecstack import stacking



models = [KNeighborsClassifier(n_neighbors=5,n_jobs=-1),

          RandomForestClassifier(random_state=0, n_jobs=-1,n_estimators=100, max_depth=3),

          XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,n_estimators=100, max_depth=3)

]

S_train, S_test = stacking(models,                   

                           X_train, y_train, X_test,   

                           regression=False, 

                           mode='oof_pred_bag', 

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score, 

                           n_folds=4, 

                           stratified=True,

                           shuffle=True,  

                           random_state=0, 

                           verbose=2)

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                      n_estimators=100, max_depth=3)

print("test accuracy: {} ".format(model.fit(S_train, y_train).score(S_test, y_test)))

print("train accuracy: {} ".format(model.fit(S_train, y_train).score(S_train, y_train)))
from keras import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

classifier = Sequential()

classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal',input_dim=8))

classifier.add(Dropout(0.2))

classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dropout(0.2))

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))



classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train,y_train, batch_size=100, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)

import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Flatten(input_shape=(8,)),

    keras.layers.Dense(16, activation=tf.nn.relu),

	keras.layers.Dense(16, activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.sigmoid),

])



model.compile(optimizer='SGD',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10, batch_size=100)



test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)




y_train= y_train.reshape((15213, 1))

y_test= y_test.reshape((2685, 1))

print(x_train.T.shape)

print(y_train.T.shape)

x_train=x_train.T

y_train=y_train.T
import numpy as np

import h5py

import matplotlib.pyplot as plt

%matplotlib inline

layers_dims =np.array([8,16,16,1])



def initialize_parameters(layer_dims):

    parameters = {}

    L = len(layer_dims)           

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01

        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

    return parameters  



def linear_forward(A, W, b):

    Z = np.dot(W,A)+b

    cache = (A, W, b)

    return Z, cache



def sigmoid(z):

    A = 1/(1 + np.exp(-z))

    activation_cache = A.copy()

    return A, activation_cache

    

def relu(z):

    A = z*(z > 0)

    activation_cache = z

    return A, activation_cache



    

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache



def L_model_forward(X, parameters):

    caches = []

    A = X

    L = len(parameters) // 2                  

    for l in range(1, L):

        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")

        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches



def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))

    cost = np.squeeze(cost)      

    return cost



def linear_backward(dZ, cache):

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,A_prev.T)

    db = (1/m)*np.sum(dZ, axis=1, keepdims = True)

    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db



def sigmoid_backward(dA, activation_cache):

    return dA*(activation_cache*(1-activation_cache))



def relu_backward(dA, activation_cache):

    return dA*(activation_cache > 0)

    

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache)        

    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)

        dA_prev, dW, db = linear_backward(dZ, linear_cache)    

    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):

    grads = {}

    L = len(caches) 

    m = AL.shape[1]

    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):

        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")

        grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l + 1)] = dW_temp

        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]

        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters



def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):

    costs = []                         

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters



def predict(X, y, parameters): 

    m = X.shape[1]

    p = np.zeros((1,m), dtype = np.int)

    a3, caches = L_model_forward(X, parameters)

    for i in range(0, a3.shape[1]):

        if a3[0,i] > 0.5:

            p[0,i] = 1

        else:

            p[0,i] = 0

    print(str(np.mean((p[0,:] == y[0,:]))))

    return p



parameters = L_layer_model(x_train, y_train, layers_dims, num_iterations = 2500, print_cost = True)

pred_train = predict(x_train, y_train, parameters)

print(pred_train)

pred_test = predict(x_test.T, y_test.T, parameters)

print(pred_test)