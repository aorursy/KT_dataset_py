import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



def plot_points(X, y):

    admitted = X[np.argwhere(y==1)]

    rejected = X[np.argwhere(y==0)]

    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k',zorder=2)

    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k',zorder=2)



def display(m, b, color='g--'):

    plt.xlim(-0.05,1.05)

    plt.ylim(-0.05,1.05)

    x = np.arange(-10, 10, 0.1)

    plt.plot(x, m*x+b, color,zorder=1)

    
data = pd.read_csv('../input/test.csv', header=None)

#((x,y) points)

X = np.array(data[[0,1]])

#point class: 1 or 0

pointType = np.array(data[2])
# Activation (sigmoid) function

def sigmoid(x):

    return 1 / (1 + np.exp(-x))



#y-hat:continuous prediction function

def output_formula(features, weights, bias):

    return sigmoid(np.dot(features, weights) + bias)



def error_formula(y, output):

    return - y*np.log(output) - (1 - y) * np.log(1-output)



def update_weights(x, y, weights, bias, learnrate):

    #y_hat

    output = output_formula(x, weights, bias)

    #y - y_hat 

    d_error = y - output

    #alfa*(y - y_hat)*x

    weights += learnrate * d_error * x

    bias += learnrate * d_error

    return weights, bias



epochs = 500

learnrate = 0.1

#setting random weights

#n_records, n_features = X.shape

#weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

#print(weights)

weights = [-1,1]

bias = 0

errors = []

last_loss = None



for e in range(epochs):

    display(-weights[0]/weights[1], -bias/weights[1])

    

    for x, y in zip(X, pointType):

        

        #print("x,y ",x,y)

        #output = output_formula(x, weights, bias)

        #print("y-hat",output)

        #error = error_formula(y, output)

        #print("error",error)

        

        weights, bias = update_weights(x, y, weights, bias, learnrate)

        

    #log-loss error on the training set

    out = output_formula(X, weights, bias)

    #print(out)

    loss = np.mean(error_formula(pointType, out))

    errors.append(loss)



    if e % (epochs / 10) == 0:

        print("\n========== Epoch", e,"==========")

        if last_loss and last_loss < loss:

            print("Train loss: ", loss, "  WARNING - Loss Increasing")

        else:

            print("Train loss: ", loss) 

        last_loss = loss



        #y-hat(output) > 0.5 => point type 1 ; < 0.5 => point type 0

        predictions = out > 0.5



        #print('y-hat=',out)

        #print('points=',pointType)

        #print('predictions=',predictions) 

        #print('verified=',predictions == pointType)



        accuracy = np.mean(predictions == pointType)

        print("Accuracy: ", accuracy)        

        

    



# Plotting the solution boundary (last generated line)

plt.title("Solution boundary")

display(-weights[0]/weights[1], -bias/weights[1], 'black')



# Plotting the data

plot_points(X, pointType)

plt.show()



# Plotting the error

plt.title("Error Plot")

plt.xlabel('Number of epochs')

plt.ylabel('Error')

plt.plot(errors)

plt.show()
#single point error

output = output_formula(x, weights, bias)

print(output)
#all individual points errors list

out = output_formula(X, weights, bias)

print(out)
#loss function: mean of all points errors

loss = np.mean(error_formula(y, out))

print(loss)