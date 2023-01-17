import matplotlib.pyplot as plt

import numpy as np



# 100 linearly spaced numbers

x = np.linspace(-0.5,3,100)



# the function, which is y = x^2 here

y = 2*(x**4) - 9*(x**3) + 10*(x**2) - x + 1



# setting the axes at the centre

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_position('zero')

ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')

ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')

ax.yaxis.set_ticks_position('left')



# plot the function

plt.plot(x,y, 'r')



# show the plot

plt.show()
def gradient_descent(gradient, x0, rate, precision, max_iterations):

    

    i = 0

    curr_x = x0

    

    while True:

        prev_x = curr_x #store current x value in prev_x

        curr_x = curr_x - rate * gradient(prev_x) #gradient descent

        diff = abs(curr_x - prev_x) #calculate the change in x

        i = i + 1 #iteration count

        print(i, ": x=", curr_x) #print iterations

        if (diff < precision) or (i == max_iterations):

            break



    return curr_x
df = lambda x: 8*(x**3) - 27*(x**2) + 20*x - 1 #the gradient of our function
min1 = gradient_descent(gradient=df, x0=0.2, rate=0.05, precision=0.000001, max_iterations=10000)    

print("The local minimum occurs at ", min1)
min2 = gradient_descent(gradient=df, x0=1.5, rate=0.01, precision=0.000001, max_iterations=10000)    

print("The local minimum occurs at ", min2)
min3 = gradient_descent(gradient=df, x0=1.001, rate=0.01, precision=0.000001, max_iterations=10000)    

print("The local minimum occurs at ", min3)
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import random
def sigmoid_activation(x):

    # compute and return the sigmoid activation value for a given input value

    return 1.0 / (1 + np.exp(-x))
dataset = pd.read_csv('../input/seeds-dataset-binary/seeds_dataset_binary.csv')



# target attribute

y = dataset['type'].values



# predictor attributes

X = dataset.drop('type', axis=1).values



# insert a column of 1's as the first entry in the feature

# vector -- this is a little trick that allows us to treat

# the bias as a trainable parameter *within* the weight matrix

# rather than an entirely separate variable



X = np.c_[np.ones((X.shape[0])), X]
# pepare independent stratified data sets for training and test of a classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, stratify=y)
def test_model(weights):

    

    accuracy = 0;

    

    for i in range(len(X_test)):

        # compute the prediction by taking the dot product of the

        # current feature vector with the weight matrix W, then

        # passing it through the sigmoid activation function

        result = sigmoid_activation(X_test[i].dot(weights))

 

        label = 0 if result < 0.5 else 1

        

        if label == y_test[i]:

            accuracy += 1

 

        # show our output classification

        # print("result={:.4f}; predicted_label={}, true_label={}".format(result, label, y_test[i]))

        

    print("[INFO] overall accuracy: ", accuracy/len(y_test))
W_init = np.random.uniform(size=(X.shape[1],))

print("W_init = ", W_init)
i = 0;

max_iterations = 10000

learning_rate = 0.01

precision = 0.0000001



# initialise the weight vector such it has the same number of elements as the number of input features

# plus 1 to account for the bias column of 1s

W = W_init



# initialize a list to store the loss value for each iteraction

loss = X_train.shape[1] * 100 # a large number

lossHistoryBGD = [loss]



# loop over the desired number of max_iterations

while True:



    # take the dot product between our features `X` and the

    # weight vector `W`, then pass this value through the

    # sigmoid activation function, thereby giving us our

    # predictions on the dataset

    predictions = sigmoid_activation(X_train.dot(W))

 

    # now that we have our predictions, we need to determine

    # our `error`, which is the difference between our predictions

    # and the true values

    errors = predictions - y_train

 

    # given our `error`, we can compute the total loss value as

    # the sum of squared loss -- ideally, our loss should

    # decrease as we continue training

    loss = np.sum(errors ** 2)

    lossHistoryBGD.append(loss)

 

    # the use of precision in this case does not help

    # instead we run the loop for a fixed number of max_iterations

    #    if  ((abs(lossHistory[-1] - lossHistory[-2]) < precision) or 

    #         (i == max_iterations) ):

    #       break



    if  i == max_iterations:

        break

        

    i=i+1

        

    # the gradient update is therefore the dot product between

    # the transpose of `X` and our error, scaled by the total

    # number of data points in `X`

    gradient = X_train.T.dot(errors) / X_train.shape[0]

 

    # in the update stage, all we need to do is nudge our weight

    # matrix in the negative direction of the gradient (hence the

    # term "gradient descent" by taking a small step towards a

    # set of "more optimal" parameters

    W += -learning_rate * gradient

    

print("[INFO] iterations #{}, loss={:.7f}, previous_loss={:.7f}".format(i, loss, lossHistoryBGD[-2]))
test_model(weights=W)
plt.plot(lossHistoryBGD[1:])

plt.ylabel('loss history')

plt.xlabel('iteration')

plt.show()
i = 0;

max_iterations = 10000

learning_rate = 0.01

precision = 0.000001

batch_size = 32



# initialise the weight vector such it has the same number of elements as the number of input features

# plus 1 to account for the bias column of 1s

W = W_init



# initialize a list to store the loss value for each iteraction

mean_batch_loss = X_train.shape[0] * 100 # a large number

lossHistorySGD = [mean_batch_loss]



def next_batch(X_train, y_train, batchSize):

    # loop over our dataset `X` in mini-batches of size `batchSize`

    for i in np.arange(0, X_train.shape[0], batchSize):

        # yield a tuple of the current batched data and labels

        yield (X_train[i:i + batchSize], y_train[i:i + batchSize])



# loop over the desired number of epochs

while True:

    

    i=i+1

    

    # initialize the total loss for the epoch

    batchLoss = []



    # loop over our data in batches

    for (batchX, batchY) in next_batch(X_train, y_train, batch_size):

        # take the dot product between our current batch of

        # features and weight matrix `W`, then pass this value

        # through the sigmoid activation function

        predictions = sigmoid_activation(batchX.dot(W))



        # now that we have our predictions, we need to determine

        # our `error`, which is the difference between our predictions

        # and the true values

        errors = predictions - batchY



        # given our `error`, we can compute the total loss value on

        # the batch as the sum of squared loss

        loss = np.sum(errors ** 2)

        batchLoss.append(loss)



        # the gradient update is therefore the dot product between

        # the transpose of our current batch and the error on the

        # # batch

        gradient = batchX.T.dot(errors) / batchX.shape[0]



        # use the gradient computed on the current batch to take

        # a "step" in the correct direction

        W += -learning_rate * gradient



    # update our loss history list by taking the average loss

    # across all batches

    

    mean_batch_loss = np.average(batchLoss)

    

    lossHistorySGD.append(mean_batch_loss)

    

    if (i == max_iterations):

        break

    

print("[INFO] iterations #{}, loss={:.7f}, previous_loss={:.7f}".format(i, mean_batch_loss, lossHistorySGD[-2]))
test_model(weights=W)
plt.plot(lossHistorySGD[1:])

plt.ylabel('loss history')

plt.xlabel('iteration')

plt.show()