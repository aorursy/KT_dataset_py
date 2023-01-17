# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt #plotting
import time #time random
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
print(train.columns.values)
test=pd.read_csv('../input/test.csv')
print(test.columns.values)
#train=np.array(train)
#np.shape(train)
#train[0]
#rang=list(range(len(set(train[:,-1]))))
#print(rang)
#rang=enumerate(rang)
#rang=list(rang)
combine=[train,test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train=train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test=test.drop(['Name','Ticket','Cabin'],axis=1)
train.head()
test.head()
train.describe()
train.describe(include=['O'])
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()
train = train.drop(['AgeBand'], axis=1)
combine = [train, test]
train.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp'], axis=1)
test = test.drop(['Parch', 'SibSp'], axis=1)

combine = [train, test]
train.head()
train['Embarked'] = train['Embarked'].fillna(train.Embarked.dropna().mode()[0])
train['Age'] = train['Age'].fillna(train.Age.dropna().median())
test['Age'] = test['Age'].fillna(test.Age.dropna().median())
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train['Sex'] = train['Sex'].map({'male':0,'female':1}).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Sex'] = test['Sex'].map({'male':0,'female':1}).astype(int)
test.head()
Train=np.array(train)
t=int(time.time())
np.random.seed(1533756006)
np.random.shuffle(Train)
X_train=Train[:,1:]
Y_train=Train[:,0]
test=np.array(test)
X_test=test[:,1:] #the first col. is the Id.
def FetNorm(X):
    #Calculate Mean, Then Std deviation for each column
    #X=X-Mean/std
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X=(X-mean)/(std)
    return X
X_train=FetNorm(X_train)
X_test=FetNorm(X_test)
print(X_test)
def create_placeholders():
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X=tf.placeholder(tf.float32,shape=(X_train.shape[1],None))
    Y=tf.placeholder(tf.float32,shape=(1,None))
    keep_prob = tf.placeholder(tf.float32)
    
    return X, Y,keep_prob
def initialize_parameters(input_shape):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [8, input_shape]
                        b1 : [8, 1]
                        W2 : [3, 8]
                        b2 : [3, 1]
                        W3 : [1, 3]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [8,input_shape], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [8,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [3,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
def forward_propagation(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.matmul(W1,X)+b1                                # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.leaky_relu(Z1)                              # A1 = relu(Z1)
    A1 = tf.nn.dropout(A1,keep_prob)                    #dropout layer.
    Z2 = tf.matmul(W2,A1)+b2                               # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.leaky_relu(Z2)                               # A2 = relu(Z2)
    A2 = tf.nn.dropout(A2,keep_prob)                    #dropout layer.
    Z3 = tf.matmul(W3,A2)+b3                               # Z3 = np.dot(W3,A2) + b3
    ### END CODE HERE ###
    
    return Z3
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###
    
    return cost
def model(X_train, Y_train, X_CV, Y_CV,X_test, learning_rate = 0.009,
          num_epochs = 4000, minibatch_size = 32,print_cost = True,th=0.56,kp=0.55):
#def model(X_train, Y_train, learning_rate = 0.01,
#         num_epochs = 6000, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples =)
    Y_train -- test set, of shape (output size = 6, number of training examples = )
    X_test -- training set, of shape (input size = 12288, number of training examples = )
    Y_test -- test set, of shape (output size = 6, number of test examples = )
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y,keep_prob = create_placeholders()
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(X_train.shape[0])
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters,keep_prob)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    threshold=tf.constant(th)
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            #num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            #seed = seed + 1
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            #for minibatch in minibatches:

                # Select a minibatch
                #(minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
            _ , ccost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train,keep_prob:kp})
                ### END CODE HERE ###
                
            epoch_cost += ccost / m

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        predicted=tf.to_float(tf.greater(tf.sigmoid(Z3),threshold))
        actual=Y
        correct_prediction = tf.equal(predicted , actual)
        TP = tf.count_nonzero(predicted * actual)
        TN = tf.count_nonzero((predicted - 1) * (actual - 1))
        FP = tf.count_nonzero(predicted * (actual - 1))
        FN = tf.count_nonzero((predicted - 1) * actual) 
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        Test=tf.to_float(tf.greater(tf.sigmoid(Z3),threshold))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        Test=Test.eval({X: X_test,keep_prob: 1.})
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train,keep_prob: 1.}))
        print ("Train F1:", f1.eval({X: X_train, Y: Y_train,keep_prob: 1.}))
        print ("CV Accuracy:", accuracy.eval({X: X_CV, Y: Y_CV,keep_prob: 1.}))
        print ("CV F1:", f1.eval({X: X_CV, Y: Y_CV,keep_prob: 1.}))

        return (f1.eval({X: X_CV, Y: Y_CV,keep_prob: 1.})/abs(f1.eval({X: X_train, Y: Y_train,keep_prob: 1.})-f1.eval({X: X_CV, Y: Y_CV,keep_prob: 1.}))),Test,parameters
middlept=885
_,Test,parameters=model(X_train.T[:,:middlept],Y_train.reshape(1,-1)[:,:middlept] \
                 ,X_train.T[:,middlept:],Y_train.reshape(1,-1)[:,middlept:],X_test.T,num_epochs = 8000,th=0.56,kp=0.9)
#f1=[]
#maxi=-1;
#maxf=0
#for i in range(50,95,5):
#    F1,Test,parameters=model(X_train.T[:,:middlept],Y_train.reshape(1,-1)[:,:middlept]  \
#             ,X_train.T[:,middlept:],Y_train.reshape(1,-1)[:,middlept:],X_test.T,kp=(i/100),num_epochs = 6000,print_cost = False)
#    f1.append(F1)
#    if(max(f1)>maxf):
 #       maxf=max(f1)
#        maxi=i
#print(maxf,maxi)
data={'PassengerID':list(test[:,0].astype(int).reshape(-1,)),'Survived':list((Test).astype(int).reshape(-1,))}
df = pd.DataFrame(data, columns = ['PassengerID', 'Survived'])
df.to_csv('sol.csv',index=False)