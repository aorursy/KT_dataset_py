import os

import numpy as np

import tensorflow as tf

import datetime

import csv as csv

import pandas as pd

import operator

print ("Packages loaded")



#load training and testing data

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]

print (train.info())

#print (test.info())





#creating not available ages

for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)





# mapping strings to ints

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)



#normalize values

def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result





train=normalize(train)

test=normalize(test)



#print (train.head(10))

#print (test.head(10))





# making np arrays out of pandas

train = train.values

test  = test.values





# Parse data

print ("Preparing train Data")

traindata = train[0::,1:train.shape[1]]

trainlabelSingle = train[0::,0:1]

trainlabel=np.zeros(shape=(trainlabelSingle.shape[0],2))

for i in range(len(trainlabelSingle)):

    if trainlabelSingle[i][0]==0:

        trainlabel[i]=[1,0]  #did not survive

    else:

        trainlabel[i]=[0,1]  #did survive





#define variables

print ("Preparing Neural Network")

tf.set_random_seed(0)

ntrain = traindata.shape[0] #number of trainingdata

n_input  = train.shape[1]-1 #number of input

n_output = 2 #number of output





#init weights and bias

weights  = {

    'wd1': tf.Variable(tf.random_normal([(int)(n_input), 128], stddev=0.1),name="wd1"),

    'wd2': tf.Variable(tf.random_normal([128, n_output], stddev=0.1),name="wd2")

}

    

biases   = {

    'bd1': tf.Variable(tf.random_normal([128], stddev=0.1),name="bd1"),

    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1),name="bd2")

}



#define network

def mlp_basic(_input, _w, _b, _keepratio):

    # make right shape

    _dense1 = tf.reshape(_input, [-1, _w['wd1'].get_shape().as_list()[0]])

    # fully connected layer 1

    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))

    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    # fully connected layer 2

    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {'out': _out}

    return out

print ("NETWORK READY")





# tf Graph input/output

x = tf.placeholder(tf.float32, [None, n_input])

y = tf.placeholder(tf.float32, [None, n_output])

keepratio = tf.placeholder(tf.float32)



# define functions

_pred = mlp_basic(x, weights, biases, keepratio)['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))

WEIGHT_DECAY_FACTOR = 0.0001

l2_loss = tf.add_n([tf.nn.l2_loss(v) 

            for v in tf.trainable_variables()])

cost = cost + WEIGHT_DECAY_FACTOR*l2_loss

optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects

accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy

init = tf.initialize_all_variables()

print ("FUNCTIONS READY")





#optimize

# Parameters

training_epochs = 10

batch_size      = 4

display_step    = 1



# Launch the graph

sess = tf.Session()

sess.run(init)



# Training cycle

print('Start time: {:[%H:%M:%S]}'.format(datetime.datetime.now()))

for epoch in range(training_epochs): 

    avg_cost = 0.

    num_batch = int(ntrain/batch_size)+1

    # Loop over all batches

    for i in range(num_batch): 

        randidx = np.random.randint(ntrain, size=batch_size)

        batch_xs = traindata[randidx,:]

        batch_ys = trainlabel[randidx,:]                

        # Fit training using batch data

        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys

                                  , keepratio:0.7})

        # Compute average loss

        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys

                                , keepratio:1.})/num_batch



    # Display logs per epoch step

    if epoch % display_step == 0 or epoch == training_epochs-1:

        print ('{:[%H:%M:%S]  }'.format(datetime.datetime.now())+"Epoch: %03d/%03d cost: %.9f" % 

               (epoch, training_epochs, avg_cost))

        train_acc = sess.run(accr, feed_dict={x: batch_xs

                                , y: batch_ys, keepratio:1.})

        print (" Training accuracy: %.3f" % (train_acc))



print ("Optimization Finished!")



#predict

predictions_file = open("output.csv", "w")

p = csv.writer(predictions_file)

p.writerow(['PassengerId', 'Survived'])



testdata = test[0::,0:test.shape[1]]



for i in range(len(testdata)):

    datapoint=testdata[i:i+1]

    predictiton=sess.run(tf.nn.softmax(_pred), feed_dict={x: datapoint,keepratio:1.}) #make prediction

    #print (predictiton)

    index, value = max(enumerate(predictiton[0]), key=operator.itemgetter(1))

    p.writerow([ntrain+1+i, index])



predictions_file.close()





sess.close()

print ("Session closed.")