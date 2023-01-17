# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#reading the data from train.csv file.

import csv

import numpy as np

data = []

with open(r'../input/train.csv','r') as train_file:

    csvRead = csv.reader(train_file)

    for row in csvRead:

        data.append(row)

        

#print data structure.

print(data[0])



#deleting first row

del data[0]



#print data size

print(len(data))



#print example data

print(data[10])



data_size = len(data)

data = np.array(data)

data[np.where(data=='')] = '0'

data[np.where(data=='nan')] = '0'



#getting right features for train_dataset (feature - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

num_features = 7

print(data.shape)

data = np.delete(data,0,1) #deleting unwanted 

data = np.delete(data,2,1)

data = np.delete(data,6,1)

data = np.delete(data,7,1)

print(data.shape)

print(data[0:5])
#making train_dataset

train_dataset = np.ndarray(shape=(data_size,num_features+1), dtype = np.float32)

train_dataset[:,0] = data[:,1].astype(np.float32)

train_dataset[:,2] = data[:,3].astype(np.float32)

train_dataset[:,3] = data[:,4].astype(np.float32)

train_dataset[:,4] = data[:,5].astype(np.float32)

train_dataset[:,5] = data[:,6].astype(np.float32)

train_dataset[:,7] = data[:,0].astype(np.float32)



sex = data[:,2]

for i,item in enumerate(sex):

    if(item=='male'):

        train_dataset[i,1] = 100

    else:

        train_dataset[i,1] = 200

embarked = data[:,7]

for i,item in enumerate(embarked):

    if(item=='C'):

        train_dataset[i,6] = 4

        continue

    if(item=='Q'):

        train_dataset[i,6] = 5

        continue

    if(item=='S'):

        train_dataset[i,6] = 6

        continue

    else:

        train_dataset[i,6] = 7

        

#Example data

print("Features - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")        

print('')

np.random.shuffle(train_dataset)

print(train_dataset[1:5])
#getting train_labels for neural network

import numpy as np

neural_train_labels = np.ndarray(shape =(data_size,2),dtype = np.float32)

for i,value in enumerate(train_dataset[:,7]):

    neural_train_labels[i,int(value)] = 1

    neural_train_labels[i,abs(int(value)-1)] = 0

print(neural_train_labels[1:5])
#getting train_labels for decison tree and logistic classifier.

train_labels = train_dataset[:,7]

train_dataset = train_dataset[:,0:7]

print(train_dataset[1:5])

print(train_labels[1:5])
#normalizing the train_dataset = (value-avg)/maxV-minV

for i in range(train_dataset.shape[1]):

    avg = train_dataset[:,i].mean()

    maxV = max(train_dataset[:,i])

    minV = min(train_dataset[:,i])

    std = np.std(train_dataset[:,i],axis=0) #standard deviation

    train_dataset[:,i] = (train_dataset[:,i]-avg)/std

print(train_dataset)
#breaking train_dataset into test_dataset and valid_dataset

test_dataset = np.ndarray(shape=(100,7),dtype=np.float32)

test_labels = np.ndarray(shape=(100,1),dtype=np.float32)

neural_test_labels = np.ndarray(shape=(100,2),dtype=np.float32)



valid_dataset = np.ndarray(shape=(100,7),dtype=np.float32)

valid_labels = np.ndarray(shape=(100,1),dtype=np.float32)

neural_valid_labels = np.ndarray(shape=(100,2),dtype=np.float32)



test_dataset = train_dataset[791:891]

test_labels = train_labels[791:891]

neural_test_labels = neural_train_labels[791:891]



valid_dataset = train_dataset[691:791]

valid_labels = train_labels[691:791]

neural_valid_labels = neural_train_labels[691:791]



train_dataset = train_dataset[0:691]

train_labels = train_labels[0:691]

neural_train_labels = neural_train_labels[0:691]



print(train_dataset.shape , train_labels.shape, neural_train_labels.shape)

print(valid_dataset.shape , valid_labels.shape, neural_valid_labels.shape)

print(test_dataset.shape , test_labels.shape, neural_test_labels.shape)
#Decision Tree Classifier.



#function for calculation of accuracy

def accuracy(X,Y):

    count = 0;

    for i in range(len(X)):

        if(X[i]==Y[i]):

            count = count+1

    return (count*100)/len(X)



#DecisionTree

from sklearn import tree

dec_tree = tree.DecisionTreeClassifier()

dec_tree.fit(train_dataset,train_labels)



dec_train_prediction = dec_tree.predict(train_dataset)

dec_valid_prediction = dec_tree.predict(valid_dataset)

dec_test_prediction = dec_tree.predict(test_dataset)





print("Training_Accuracy = %d%%" %accuracy(dec_train_prediction,train_labels))

print("Validation_Accuracy = %d%%" %accuracy(dec_valid_prediction,valid_labels))

print("Test_Accuracy = %d%%" %accuracy(dec_test_prediction,test_labels))
#Logistic Regression

def accuracy(X,Y):

    count = 0;

    for i in range(len(X)):

        if(X[i]==Y[i]):

            count = count+1

    return (count*100)/len(X)



from sklearn import linear_model

log_reg = linear_model.LogisticRegression(penalty='l2',solver ='newton-cg',tol = 0.0001,C=1000, max_iter=10000)

log_reg.fit(train_dataset,train_labels)



log_train_prediction = log_reg.predict(train_dataset)

log_valid_prediction = log_reg.predict(valid_dataset)

log_test_prediciton = log_reg.predict(test_dataset)



print("Training_accuracy = %d%%" %(log_reg.score(train_dataset,train_labels)*100))

print("Valid_accuracy = %d%%" %(log_reg.score(valid_dataset,valid_labels)*100))

print("Test_accuracy = %d%%" %(log_reg.score(test_dataset,test_labels)*100))
import tensorflow as tf

data_size = train_dataset.shape[0]

output_size = 2

hidden_layer1 = 32

hidden_layer2 = 32

beta = 0.005

start_learning_rate = 0.03



graph = tf.Graph()



with graph.as_default():

    tf_train_dataset = tf.constant(train_dataset)

    tf_valid_dataset = tf.constant(valid_dataset)

    tf_test_dataset = tf.constant(test_dataset)

    tf_train_labels = tf.constant(neural_train_labels)



    layer1_weights = tf.Variable(tf.truncated_normal([7,hidden_layer1],stddev=0.1))

    layer1_biases = tf.Variable(tf.truncated_normal([hidden_layer1],stddev=0.01))

    layer2_weights = tf.Variable(tf.truncated_normal([hidden_layer1,hidden_layer2],stddev=0.1))

    layer2_biases = tf.Variable(tf.truncated_normal([hidden_layer2],stddev=0.01))

    layer3_weights = tf.Variable(tf.truncated_normal([hidden_layer2,output_size],stddev=0.1))

    layer3_biases = tf.Variable(tf.truncated_normal([output_size],stddev=0.01))

    

    logits = tf.matmul(tf_train_dataset,layer1_weights) + layer1_biases

    logits = tf.nn.relu(logits)

    logits = tf.matmul(logits,layer2_weights) + layer2_biases

    logits = tf.nn.relu(logits)

    logits = tf.matmul(logits,layer3_weights) + layer3_biases



    regularizer = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights)# + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer5_weights)



    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=neural_train_labels, logits=logits))

    loss = tf.reduce_mean(loss + regularizer*beta)



    global_step = tf.Variable(0)  # count the number of steps taken.

    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 20000, 0.95, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    

    neural_train_prediction = tf.nn.softmax(logits)

    neural_valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset,layer1_weights) + layer1_biases), layer2_weights)+layer2_biases),layer3_weights) +layer3_biases)

    neural_test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset,layer1_weights) + layer1_biases), layer2_weights)+ layer2_biases),layer3_weights)+layer3_biases)

    
def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])
num_steps = 20001

final_weights1 = []

final_biases1 = []

final_weights2 = []

final_biases2 = []

final_weights3 = []

final_biases3 = []



neural_train_predictions = []

neural_valid_predictions = []

neural_test_predictions = []



J = []



with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()

    print("Initialized")

    for step in range(num_steps):

        feed_dict = {tf_train_dataset : train_dataset, tf_train_labels : neural_train_labels}

        _, l, predictions,final_weights1,final_biases1,final_weights2,final_biases2,final_weights3,final_biases3 = session.run([optimizer, loss, neural_train_prediction,layer1_weights,layer1_biases,layer2_weights,layer2_biases,layer3_weights,layer3_biases], feed_dict=feed_dict)

        J.append(l)

        if(step%2000==0):

            print("Step : %d"%step)

            print("Training Loss : %f"%l)

            print("Training Accuracy : %d%%" %accuracy(predictions,neural_train_labels))

    neural_train_predictions = neural_train_prediction.eval()

    neural_valid_predictions = neural_valid_prediction.eval()

    neural_test_predictions = neural_test_prediction.eval()

    print("Validation Accuracy : %d%%" %accuracy(neural_valid_prediction.eval(),neural_valid_labels))

    print("Test Accuracy : %d%%" %accuracy(neural_test_prediction.eval(),neural_test_labels))
import matplotlib.pyplot as plt

it = list(range(0,num_steps))

plt.plot(it[1000:],J[1000:])

plt.xlabel("Iterations")

plt.ylabel("Error")

plt.title("Error Graph")

plt.show()