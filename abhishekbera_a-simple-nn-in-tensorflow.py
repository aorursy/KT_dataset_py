#from helpers.clean_titanic import load_dataset

import tensorflow as tf

import numpy as np

from pandas import read_csv

import pandas as pd

from matplotlib import pyplot as plt
def impute_age(titanic_data):

    """

    Input: ages_missing (dataframe) having keys ("Survived", "names", "Embarked", "ages", "Pclass")

    outputs : array containing all the ages including the imputed data

    """

    for i in np.squeeze(np.where(pd.isnull(titanic_data["Age"]))):

        prefix = ""

        if "Mr" in titanic_data.Name[i]:

            prefix = "Mr"

        elif "Mrs" in titanic_data.Name[i]:

            prefix = "Mrs"

        elif "Miss" in titanic_data.Name[i]:

            prefix = "Miss"

        else:

            prefix = "Master"

        titanic_subset = titanic_data[(titanic_data.Embarked == titanic_data.Embarked[i])

                                      & (titanic_data.Pclass == titanic_data.Pclass[i]) &

                                     (titanic_data.Survived == titanic_data.Survived[i])&

                                      (titanic_data.Sex == titanic_data.Sex[i] ) &

                                      (titanic_data.Name.str.contains(prefix))

                                     ]

        titanic_data.Age[i] = titanic_subset.Age.median()

    return titanic_data
def impute_data(titanic_data):

    titanic_data["Embarked"] = titanic_data["Embarked"].fillna(value = "S")

    titanic_data = impute_age(titanic_data)

    titanic_data["Name"] = titanic_data["Name"].fillna(value = "")

    titanic_data["Ticket"] = titanic_data["Ticket"].fillna(value = "")

    titanic_data["Cabin"] = titanic_data["Cabin"].fillna(value = "")

    return titanic_data
def load_dataset(sample_size = 500):

    """

    returns the features and labels of training and testing set indivisually

    """

    #read the data

    train_data = read_csv("../input/titanic/train.csv")

    train_data = impute_data(train_data)

    #train_data = train_data.sample(frac = 1).reset_index(drop = True)

    train_data = train_data.drop(axis = 1, labels = ["Name","Ticket","Cabin"]).dropna()

    #setting data into indivisual arrays

    embarked = train_data["Embarked"].replace(["S","C","Q"], [1,2,3]).as_matrix()

    sex = train_data["Sex"].replace(["male","female"], [0,1]).as_matrix()

    pclass = train_data["Pclass"].as_matrix()

    age = train_data["Age"].as_matrix()

    sibsp = train_data["SibSp"].as_matrix()

    parch = train_data["Parch"].as_matrix()

    fare = train_data["Fare"].as_matrix()

    survived = train_data["Survived"].tolist()

    

    #sample space size

    sample_size = 500

    #create a feature vector and labels for training set

    train_features = {

        "pclass" : pclass[:sample_size],

        "age" : age[:sample_size],

        "sex" : sex[:sample_size],

        "sibsp" : sibsp[:sample_size],

        "parch" : parch[:sample_size],

        "fare" : fare[:sample_size],

        "embarked":embarked[:sample_size]

    }

    train_labels = survived[:sample_size]

    

    #create a feature vector and labels for test set

    test_features = {

         "pclass" : pclass[sample_size:],

         "age" : age[sample_size:],

         "sex" : sex[sample_size:],

         "sibsp" : sibsp[sample_size:],

         "parch" : parch[sample_size:],

         "fare" : fare[sample_size:],

        "embarked": embarked[sample_size:]

    }

    

    test_labels = survived[sample_size:]

    return train_features, train_labels, test_features, test_labels, sample_size, train_data.shape[0]



train_features, train_labels, test_features, test_labels, m, t = load_dataset()

train_features = np.array(

    [train_features["pclass"],

    train_features["age"],

    train_features["sex"],

    train_features["sibsp"],

    train_features["parch"],

    train_features["fare"],

    train_features["embarked"]

    ]

)



train_labels = np.array(train_labels)



test_features = np.array(

    [test_features["pclass"],

    test_features["age"],

    test_features["sex"],

    test_features["sibsp"],

    test_features["parch"],

    test_features["fare"],

    test_features["embarked"]

    ]

)



test_labels = np.array(test_labels)

assert train_features.shape == (7,m)

#assert train_labels.shape == (1,m)

assert test_features.shape == (7, t - m)

#assert test_labels.shape == (1, t - m)
def one_hot_encoder(labels, C):

    C = tf.constant(C, dtype = tf.int32)

    one_hot_matrix = tf.one_hot(labels, depth = C, axis = 0)

    session = tf.Session()

    one_hot_mat = session.run(one_hot_matrix)

    session.close()

    return one_hot_mat

    
train_labels = one_hot_encoder(train_labels, 2)

test_labels = one_hot_encoder(test_labels, 2)

print(test_labels.shape)
def create_placeholders(n_x, n_y):

    X = tf.placeholder(dtype = tf.float32, shape = [n_x, None])

    Y = tf.placeholder(dtype = tf.float32, shape = [n_y, None])

    

    return X, Y
tf.set_random_seed(1)

def initialize_network(scalea, scaleb):

    #W1 = tf.get_variable("W1", [8,7], initializer = tf.contrib.layers.xavier_initializer())

    W1 = tf.get_variable("W1", [8,7], initializer = tf.contrib.layers.xavier_initializer(seed = 4), regularizer=tf.contrib.layers.l2_regularizer(scale=scalea))

    b1 = tf.get_variable("b1", [8,1], initializer = tf.zeros_initializer())

    #W2 = tf.get_variable("W2", [12,8], initializer = tf.contrib.layers.xavier_initializer())

    W2 = tf.get_variable("W2", [12,8], initializer = tf.contrib.layers.xavier_initializer(seed = 4), regularizer=tf.contrib.layers.l2_regularizer(scale=scaleb))

    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [2,12], initializer = tf.contrib.layers.xavier_initializer(seed = 4))

    b3 = tf.get_variable("b3", [2,1], initializer = tf.zeros_initializer())

    parameters = {

        "W1": W1,

        "b1": b1,

        "W2": W2,

        "b2": b2,

        "W3": W3,

        "b3": b3

    }

    return parameters
def forward_propagation(X, parameters):

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    W3 = parameters["W3"]

    b3 = parameters["b3"]

    

    Z1 = tf.add(tf.matmul(W1,X), b1)

    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)

    A2 = tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)

    

    return Z3 
def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost
def model(X_train, Y_train, X_test, Y_test, scalea = 0.5, scaleb = 0.5, learning_rate = 0.005, num_iterations = 10000, print_cost = True):

    tf.reset_default_graph()

    X, Y = create_placeholders(7,2)

    parameters = initialize_network(scalea, scaleb)

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    costs = []

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()

    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        #writer = tf.summary.FileWriter("./my_graph", graph = sess.graph)

        each_cost = 0

        for i in range(num_iterations):

            _, each_cost = sess.run([optimizer, cost], feed_dict = {X: X_train, Y:Y_train})

            if print_cost and i%5000 == 0:

                print("Cost at %i is %f"%(i, each_cost))

            if i % 10 == 0:

                costs.append(each_cost)

        

        #Metrics business

        output = tf.argmax(Z3)

        labels = tf.argmax(Y)

        correct_prediction = tf.equal(output, labels)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

       

        #training and testing accuracy

        train_accuracy = accuracy.eval({X: X_train, Y:Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        

        #precision and recall

        precisions = tf.metrics.precision(labels, output)

        recalls = tf.metrics.recall(labels, output)

        

        sess.run(tf.local_variables_initializer())

        precisions = sess.run(precisions, feed_dict={X: X_test, Y: Y_test})[1]

        recalls = sess.run(recalls, feed_dict={X: X_test, Y: Y_test})[1]

        f1_score = 2 * precisions * recalls / (precisions + recalls)

        

        metrics = {

            "train_accuracy": train_accuracy,

            "test_accuracy": test_accuracy,

            "precision": precisions,

            "recall": recalls,

            "f1_score": f1_score

        }

        return metrics, costs, sess.run(parameters)
metrics, costs, parameters = model(train_features, train_labels, test_features, test_labels, scalea = 0.26, scaleb = 0.73, num_iterations=33000, learning_rate = 0.0005, print_cost = False)

print(metrics)
plt.plot(costs)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.ylim(0.35, 0.8)

plt.title("learning_rate = 0.0005")

plt.show()