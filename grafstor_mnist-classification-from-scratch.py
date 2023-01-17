#for mathematical calculations

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
data_path = '/kaggle/input/digit-recognizer'



train = pd.read_csv(data_path+"/train.csv")

test = pd.read_csv(data_path+"/test.csv")



train_x = train.drop(labels=["label"], axis=1)

train_y = train["label"]
test = np.array(test[:11])

test =  test / 255

train_x = np.array(train_x)

train_x = train_x /255
train_X, test_X, train_Y, test_Y = train_test_split(train_x,

                                                    train_y,

                                                    test_size=0.2,

                                                    random_state=42)
train_X = np.array(train_X)

test_X = np.array(test_X)



train_Y = np.array(train_Y)

test_Y = np.array(test_Y)
# one hot encoding

def vectorize(sequence, maxlen):

    vectors = np.zeros((len(sequence), maxlen))

    for i, nums in enumerate(sequence):

        vectors[i, nums] = 1

    return vectors
train_Y = vectorize(train_Y, 10)

test_Y = vectorize(test_Y, 10)
def softmax(x):

    exps = np.exp(x - np.max(x, axis=1, keepdims=True))

    return exps/np.sum(exps, axis=1, keepdims=True)



def sigmoid(x):

    return 1 / (1 + np.exp(-x))

        

def sigmoid_derivative(x):

    return x*(1 - x)
def dropout(layer, dropout_prop):

    rs = np.random.RandomState(123)

    prop = 1 - dropout_prop

    mask = rs.binomial(size=layer.shape, n=1, p=prop)

    layer *= mask/prop

    return layer
def acc(y, p):

    accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(p, axis=1)) / len(y)

    return accuracy
weights_1 = 2*np.random.random((784, 784)) - 1

weights_2 = 2*np.random.random((784, 10)) - 1



bias_1 = 0

bias_2 = 0



epochs = 100

dropout_prop = 0.5

learning_rate = 0.0001

branch_size = 512



history = {

    'train': [],

    'test': []

    }
for epoch  in range(epochs):



    for branch in range(len(train_X)//branch_size+1):

        # branch data

        X = train_X[branch*branch_size:(branch+1)*branch_size]

        Y = train_Y[branch*branch_size:(branch+1)*branch_size]

        

        ### FEED FORWARD

        l1 = sigmoid(np.dot(X, weights_1)) + bias_1

        l2 = softmax(np.dot(l1, weights_2)) + bias_2

        l2 = dropout(l2, dropout_prop)



        ### BACK PROPOGATION



        # first layer error

        delta = l2 - Y



        w_adjustment_1 = np.dot(l1.T, delta)

        b_adjustment_1 = np.sum(delta, axis=0, keepdims=True)



        # second layer error

        error = np.dot(delta, weights_2.T)

        delta = error*sigmoid_derivative(l1) 



        w_adjustment_2 = np.dot(X.T, delta)

        b_adjustment_2 = np.sum(delta, axis=0)





        # change weights

        # first adjustment

        weights_1 -= learning_rate * w_adjustment_2

        bias_1 -= learning_rate * b_adjustment_2

        

        # second adjustment

        weights_2 -= learning_rate * w_adjustment_1

        bias_2 -= learning_rate * b_adjustment_1

    

    ### TEST ACCURACY

    

    # make prediction with train data

    l1 = sigmoid(np.dot(train_X, weights_1)) + bias_1

    train_P = softmax(np.dot(l1, weights_2)) + bias_2

    

    # accuracy of train data 

    train_acc = round(acc(train_Y, train_P), 4)

    history['train'].append(train_acc)



    # make prediction with validation data

    l1 = sigmoid(np.dot(test_X, weights_1)) + bias_1

    test_P = softmax(np.dot(l1, weights_2)) + bias_2



    # accuracy of test data

    test_acc = round(acc(test_Y, test_P), 4)

    history['test'].append(test_acc)

    

    # view error

    error_line = f'Epoch: {epoch+1}  Train  [acc: {train_acc}]  Validation  [acc: {test_acc}]'

    print(error_line)
plt.plot(history['test'], label='test')

plt.plot(history['train'], label='train')



plt.xlabel('epoch')

plt.ylabel('true answers')

plt.legend()

plt.show()
i = 5

test_case = test[i:i+1]



# make prediction with test case

l1 = sigmoid(np.dot(test_case, weights_1)) + bias_1

l2 = softmax(np.dot(l1, weights_2)) + bias_2



# display prediction

plt.bar([str(i) for i in range(0,10)], l2[0])

plt.xlabel("numbers")

plt.ylabel("probability")

plt.show()



# display test case number

test_case = test_case[0]

test_case = test_case.reshape(28,28)

test_case *= 255



plt.imshow(test_case)

plt.show()