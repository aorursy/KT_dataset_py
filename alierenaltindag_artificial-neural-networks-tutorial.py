import numpy as np # Linear algebra.

import pandas as pd # Data processing.

import matplotlib.pyplot as plt # Visualize



from sklearn.model_selection import train_test_split # For data split.

from sklearn.model_selection import cross_val_score # For find accuracy.



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/data.csv")

data.drop(["id"],axis = 1,inplace = True)



data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] # M = 1, B = 0



x_data = data.drop(["diagnosis","Unnamed: 32"],axis = 1)



x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values # Normalize data

y = data.diagnosis
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.15, random_state = 42) # 85% Train, 15% Test



x_train = x_train.values.T

x_test = x_test.values.T

y_test = y_test.values.reshape(1,y_test.shape[0])

y_train = y_train.values.reshape(1,y_train.shape[0])
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0

    return w, b
def sigmoid(z):

    y_head = 1/(1+np.exp(-z)) # It is the formule of sigmoid function

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    y_head = sigmoid(np.dot(w.T,x_train) + b) # We multiply features with our weight values, add bias and send it to sigmoid function

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) # It is the formule of loss function

    cost = (np.sum(loss))/x_train.shape[1] # Calculate cost function

    

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1] # Calculate derivative of weights

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] # Calculate derivative of bias

    

    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []



    for i in range(number_of_iterarion):

        # Start learning

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train) # Make forward and backward propagation and calculate cost and derivatives

        cost_list.append(cost)

        # Updating weight and bias

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 250 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))



    parameters = {"w": w,"b": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def prediction(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b) # z -> Estimates of our model

    y_prediction = np.zeros((1,x_test.shape[1])) # We create an array, we will set the array.

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1



    return y_prediction
def artificial_neural_networks(x_train,x_test,y_train,y_test,learning_rate,number_of_iteration):

    dimension = x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension) # Initialize Parameters

    parameters,gradients, cost_list = update(w,b,x_train,y_train,learning_rate,number_of_iteration) # Update parameters

    

    train_prediction = prediction(parameters["w"],parameters["b"],x_train) # Estimates of our model

    test_prediction = prediction(parameters["w"],parameters["b"],x_test) # Estimates of our model

    

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(train_prediction - y_train)) * 100))

    print("Test accuracy: {} %".format(100 - np.mean(np.abs(test_prediction - y_test)) * 100))
learning_rate = 1

number_of_iteration = 5000

ann = artificial_neural_networks(x_train,x_test,y_train,y_test,learning_rate,number_of_iteration)
from sklearn.model_selection import cross_val_score

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense
def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation = "tanh", input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation = "tanh"))

    classifier.add(Dense(units = 2, kernel_initializer = "uniform", activation = "tanh"))

    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

    classifier.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

    return classifier
x_train = x_train.T

y_train = y_train.T



classifier = KerasClassifier(build_fn = build_classifier, epochs = 100) # epoch -> Number of Iteration

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3) # Cross validation score

mean = accuracies.mean()
print(accuracies)

print("Accuracy mean :",mean)