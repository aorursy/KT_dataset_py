

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.python.framework import ops

import matplotlib.pyplot as plt
def read_data(file_name):

    """ Read training/test data """

    data = pd.read_csv('../input/'+file_name+'.csv')

    return data
def prepare_age(data):

    """Fill the missing data (nan) with average age """

    age = data['Age']

    mean_age = age.mean()

    var_age = age.var()

    age[age.isnull()] = mean_age

    age = age - mean_age

    age = age / var_age

    #print("Age is "+str(age))

    return age.as_matrix()
def prepare_fare(data):

    """ Read Fare Data and Normalize it """

    fare = data['Fare']

    mean_fare = fare.mean()

    var_fare = fare.var()

    fare = fare - mean_fare

    fare = fare / var_fare

    return fare.as_matrix()
def prepare_sex(data):

    """ Transform the male into 0 and female into 1 """

    sex = data['Sex']

    sex = np.where(sex=='male',0,1)

    #print("sex is "+str(sex))

    return sex
def prepare_embarquation(data):

    """ Transforms the embarquation """

    embarked = data['Embarked']

    embarked[embarked.isnull()] = 3

    embarked = np.where(embarked=='C', 0, embarked)

    embarked = np.where(embarked=='Q', 1, embarked)

    embarked = np.where(embarked=='S', 2, embarked)

    #Normalize

    mean_embarked = embarked.mean()

    var_embarked = embarked.var()

    embarked = embarked - mean_embarked

    embarked = embarked / var_embarked

    

    return embarked
def prepare_sibligs(data):

    """ Prepare the SibSp data """

    sib = data['SibSp']

    mean_sib = sib.mean()

    var_sib = sib.var()

    sib = sib - mean_sib

    sib = sib / var_sib

    return sib
def prepare_parch(data):

    """ Prepare the Parch data """

    parch = data['Parch']

    mean_parch = parch.mean()

    var_parch = parch.var()

    parch = parch - mean_parch

    parch = parch / var_parch

    return parch
def prepare_family_size(data):

    """ Merges the SibSp and Parch into family size then normalize """

    parch = data['Parch']

    sib = data['SibSp']

    #merge

    family_size = parch + sib

    

    #normalize

    mean_family_size = family_size.mean()

    var_family_size = family_size.var()

    family_size = family_size - mean_family_size

    family_size = family_size / var_family_size

    return family_size

    
def normalize_features(data):

    """ Normalize the features from the data """  

    age = prepare_age(data)

    sex = prepare_sex(data)

    embark = prepare_embarquation(data)

    fare = prepare_fare(data)

    sib = prepare_sibligs(data)

    parch = prepare_parch(data)

    family_size = prepare_family_size(data)

    X_train = np.column_stack((sex, age, family_size, embark))

    return X_train.T

    
def prepare_training_data():

    """ Read Training data and clean it """

    pd.set_option('mode.chained_assignment', None)

    data = read_data('train')

    X_train = normalize_features(data)

    Y_train = np.reshape(data['Survived'].as_matrix(), (X_train.shape[1],1)).T



    return X_train, Y_train
def prepare_test_data():

    """ Read Test data and clean it """

    pd.set_option('mode.chained_assignment', None)

    data = read_data('test')

    X_test = normalize_features(data)

    return X_test
def initialize_Parameters(nb_features):

    """Init parameters of W and b: 3 layers with:

    l = index of the layer

    W[l] = (n[l], n[l-1])

    b[l] = (n[l], 1)

    dW[l] = (n[l], n[l-1])

    db[l] = (n[l], 1)

    We will try the following:

           O

        O  O

        O  O  O

    X   O  O  O  O Y_hat

        O  O  O

           O  O

           O  O

           

    n[0] = nb_features

    n[1] = 5 5 

    n[2] = 7 8 

    n[3] = 6 6

    n[4] = 1 4

    n[5] = 1

    """

    W1 = tf.get_variable("W1", [5, nb_features], initializer = tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable("b1", [5,1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [8,5], initializer = tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable("b2", [8,1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [6,8], initializer = tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

    W4 = tf.get_variable("W4", [4,6], initializer = tf.contrib.layers.xavier_initializer())

    b4 = tf.get_variable("b4", [4,1], initializer = tf.zeros_initializer())

    W5 = tf.get_variable("W5", [1,4], initializer = tf.contrib.layers.xavier_initializer())

    b5 = tf.get_variable("b5", [1,1], initializer = tf.zeros_initializer())

    

    parameters = {"W1": W1, "b1": b1,

                  "W2": W2, "b2": b2,

                  "W3": W3, "b3": b3,

                  "W4": W4, "b4": b4,

                  "W5": W5, "b5": b5

                 }

    return parameters



def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"

                  the shapes are given in initialize_parameters



    Returns:

    Z4 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    W4 = parameters['W4']

    b4 = parameters['b4']

    W5 = parameters['W5']

    b5 = parameters['b5']

    

    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2

    A2 = tf.nn.relu(Z2)                                    # A1 = relu(Z1)

    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z2 = np.dot(W2, a1) + b2

    A3 = tf.nn.relu(Z3)                                    # A1 = relu(Z1)

    Z4 = tf.add(tf.matmul(W4, A3), b4) 

    A4 = tf.nn.relu(Z4)                                    # A1 = relu(Z1)

    Z5 = tf.add(tf.matmul(W5, A4), b5) # Z2 = np.dot(W2, a1) + b2



    return Z5
def create_placeholders(n_x, n_y):

    """

    Creates the placeholders for the tensorflow session.

    

    Arguments:

    n_x -- scalar, size of an titanic passenger entry 

    n_y -- scalar, number of output, here survived or not 

    

    Returns:

    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"

    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    

    Tips:

    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.

      In fact, the number of examples during test/train is different.

    """

    X = tf.placeholder(dtype=tf.float32, shape=([n_x, None]), name="X")

    Y = tf.placeholder(dtype=tf.float32, shape=([n_y, None]), name="Y")

    

    return X, Y
def compute_cost(Z, Y):

    """

    Computes the cost

    

    Arguments:

    Z4 -- output of forward propagation 

    Y -- "true" labels vector placeholder, same shape as Z4

    

    Returns:

    cost - Tensor of the cost function

    """

    

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

    logits = tf.transpose(Z)

    labels = tf.transpose(Y)

    #cost = tf.reduce_mean(tf.squared_difference(tf.sigmoid(logits), labels))

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    

    return cost
def plot_cost(costs, learning_rate):

    """ Plots the costs """

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()
def train_predict_model(learning_rate, epoch, X_train, Y_train, X_test):

    """ Training Model """



    #create placeholders for TF for X and Y

    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])



    #initialize parameters

    parameters = initialize_Parameters(X_train.shape[0])

    #compute Forward Propagation

    Z4 = forward_propagation(X, parameters)



    #compute Cost: Forward Propagation

    cost = compute_cost(Z4, Y)



    #Backward Propagation

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    

    # To keep track of the cost

    costs = []



    #print("X_train "+str(X_train))

    init = tf.global_variables_initializer()



    with tf.Session() as sess:

        sess.run(init)

        for i in range(1,epoch):        

            #feed with the training sets

            _ , calculated_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            # Print the cost every 

            #if i % 100 == 0:

                #print ("Cost after iteration %i: %f" % (i, calculated_cost))

            if i % 50 == 0:

                costs.append(calculated_cost)

        

        # plot the cost

        plot_cost(costs, learning_rate)



        # save the parameters in a variable

        parameters = sess.run(parameters)

        #print ("Parameters have been trained!")



        y_pred_proba = tf.cast(tf.nn.sigmoid(Z4),dtype = tf.float32);

        y_pred_class = tf.cast(tf.greater(y_pred_proba, 0.5),'float')

        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, Y ), 'float'))

    

        prediction=tf.argmax(tf.nn.sigmoid(Z4))

        sess.run([prediction],  feed_dict={X: X_train, Y: Y_train})

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        

        prediction_test = sess.run(y_pred_class, feed_dict={X:X_test})

        #print("Predicted :"+str(prediction_Test))

        

        return prediction_test

    
#Main

ops.reset_default_graph() 

from subprocess import check_output

#clean training Set

X_train, Y_train = prepare_training_data()

#clean test set

X_test = prepare_test_data()



#train and predict

prediction_test = train_predict_model(learning_rate=0.0001, epoch=40000, X_train= X_train

                                 , Y_train = Y_train, X_test = X_test)



#transform values to binary

prediction_test = np.where( prediction_test < 1, 0, 1) 

# prepare the output submission

data = read_data('test')

submission = pd.DataFrame({

        "PassengerId": data["PassengerId"],

        "Survived": prediction_test.reshape(X_test.shape[1])

    })



submission.to_csv('submission.csv', index=False)

#print(str(submission))

print(check_output(["ls", "."]).decode("utf8"))


