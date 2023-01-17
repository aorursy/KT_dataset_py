import pandas as pd

train_set=pd.read_csv("../input/train.csv")  # read the CSV file
train_set.head() #show the first few rows


import matplotlib.pyplot as plt  # For graphics

train_set["label"].hist()    # This is actually a pandas function
plt.show()                   # Show the plot! 
import tensorflow as tf

tf.reset_default_graph()                     # to avoid error message in the notebook. Probably you will never use it in a real script

##### Prepare data with Pandas ######
X_train=train_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_train=X_train.values                          # only the numbers, no column labels 
y_train=train_set["label"]                      # Keep only column "label"
y_train=y_train.values                          # only the numbers, no column labels 

###### TensorFlow starts here  ######
# Define the variables
X=tf.constant(X_train,dtype=tf.float32,name="X")  #input ..... Why float?!
y=tf.constant(y_train,dtype=tf.int64,name="y")    # Correct output: Not prediction, but correct answer!

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")
    
##### Grading ######
# For human beings : Accuracy
with tf.name_scope("Accuracy"):
    Guess=tf.nn.in_top_k(output_layer,y,1)    # Pick the digit with highest probability
    accuracy=tf.reduce_mean(tf.cast(Guess, tf.float32))  # Count how many you got right and give %

# For the neuronal network: Loss Function
with tf.name_scope("LossFunction"):
    EntropyProbability=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer) # compute the cross entropy
    loss=tf.reduce_mean(EntropyProbability,name="Loss") # compute average cross entropy

with tf.name_scope("Revision"):
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    training_op=optimizer.minimize(loss)

import glob                                # To check if files exist. See below

init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)

n_epochs=50                              # how many "training sessions"
save_file="./model.ckpt"                 # checkpoint. Better have one.

with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # If we have run this before, just continue 
        print("Parameters Found. Continuing")
        saver.restore(sess,save_file)    
    else:
        print("New Optimization Started")
        init.run()                       # If it is the first time, initialize.
        
    for i in range(n_epochs):
        sess.run(training_op)   # this is a training session and how to train it (training_op is defined above)
        if i % 10==0:
            print("Epoch:",i,"Loss:",loss.eval(),"accuracy:",accuracy.eval())  # notice the eval()!!
        
    print("Final Loss:",loss.eval(),"Final Accuracy",accuracy.eval())                 # Best result
    save_path = saver.save(sess, save_file)     # Save stuff for later. It returns a path ... 
    print("Model data saved in %s" % save_path)      # ... and we can well print it!
    
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    

from sklearn.preprocessing import StandardScaler

##### Prepare data with Pandas ######
## See code above ##

#### Normalize Data with scikit ####
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

###### TensorFlow starts here  ######
## See code above ##

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import  glob

tf.reset_default_graph()                     # to avoid error message in the notebook. Probably you will never use it in a real script
train_set=pd.read_csv("../input/train.csv")  # read the CSV file

##### Prepare data with Pandas ######
X_train=train_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_train=X_train.values                          # only the numbers, no column labels 
y_train=train_set["label"]                      # Keep only column "label"
y_train=y_train.values                          # only the numbers, no column labels 

#### Normalize Data with scikit ####
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

###### TensorFlow starts here  ######
# Define the variables
X=tf.constant(X_train,dtype=tf.float32,name="X")  #input ..... Why float?!
y=tf.constant(y_train,dtype=tf.int64,name="y")    # Correct output: Not prediction, but correct answer!

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")

##### Grading ######
# For human beings : Accuracy
with tf.name_scope("Accuracy"):
    Guess=tf.nn.in_top_k(output_layer,y,1)    # Pick the digit with highest probability
    accuracy=tf.reduce_mean(tf.cast(Guess, tf.float32))  # Count how many you got right and give %

# For the neuronal network: Loss Function
with tf.name_scope("LossFunction"):
    EntropyProbability=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer) # compute the cross entropy
    loss=tf.reduce_mean(EntropyProbability,name="Loss") # compute average cross entropy

with tf.name_scope("Revision"):
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    training_op=optimizer.minimize(loss)

##### Lets run the neuronal network!  #####

init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)

n_epochs=50                              # how many "training sessions"
save_file="./modelNormalized.ckpt"                 # checkpoint. Better have one.

with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # If we have run this before, just continue 
        print("Parameters Found. Continuing")
        saver.restore(sess,save_file)    
    else:
        print("New Optimization Started")
        init.run()                       # If it is the first time, initialize.
        
    for i in range(n_epochs):
        sess.run(training_op)   # this is a training session and how to train it (training_op is defined above)
        if i % 10==0:
            print("Epoch:",i,"Loss:",loss.eval(),"accuracy:",accuracy.eval())  # notice the eval()!!
        
    print("Final Loss:",loss.eval(),"Final Accuracy",accuracy.eval())                 # Best result
    save_path = saver.save(sess, save_file)     # Save stuff for later. It returns a path ... 
    print("Model data saved in %s" % save_path)      # ... and we can well print it!
    
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import  glob
import sys

tf.reset_default_graph()                     # to avoid error message in the notebook. Probably you will never use it in a real script
test_set=pd.read_csv("../input/test.csv")  # read the CSV file

##### Prepare data with Pandas ######
#X_test=test_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_test=test_set.values                          # only the numbers, no column labels 

#### Normalize Data with scikit ####
scaler=StandardScaler()
scaler.fit(X_test)
X_test=scaler.transform(X_test)

###### TensorFlow starts here  ######
# Define the variables
X=tf.constant(X_test,dtype=tf.float32,name="X") 

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")

##### Lets run the neuronal network!  #####
init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)
n_epochs=50                              # how many "training sessions"
save_file="./modelNormalized.ckpt"                 # checkpoint. Better have one.

with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # This time you MUST have a trained NN
        saver.restore(sess,save_file)    
    else:
        sys.exit('I need a trained set of parameters!')    
    prediction=np.argmax(output_layer.eval(),axis=1)
    # REMEMBER to commit the notebook to save the output file!
    #Here we create the file for the output. It has to be CSV, so let's use pandas
    d={'ImageId': np.arange(1,1+X_test.shape[0]),'label': prediction }
    df=pd.DataFrame(data=d)
    df.to_csv('submission.csv',index=None)   # print data into csv. Ready for submission!
        
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    

# Define the variables
X=tf.placeholder(shape=(None,28*28),dtype=tf.float32,name="X")  # training set 
y=tf.placeholder(shape=(None),dtype=tf.int64,name="y")    # Correct output: Not prediction, but correct answer!

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")

##### Grading ######
# For human beings : Accuracy
with tf.name_scope("Accuracy"):
    Guess=tf.nn.in_top_k(output_layer,y,1)    # Pick the digit with highest probability
    accuracy=tf.reduce_mean(tf.cast(Guess, tf.float32))  # Count how many you got right and give %

# For the neuronal network: Loss Function
with tf.name_scope("LossFunction"):
    EntropyProbability=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer) # compute the cross entropy
    loss=tf.reduce_mean(EntropyProbability,name="Loss") # compute average cross entropy

with tf.name_scope("Revision"):
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    training_op=optimizer.minimize(loss)


import numpy as np                      # for math
##### Lets run the neuronal network!  #####

init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)

n_epochs=50                              # how many "training sessions"
save_file="./modelMB.ckpt"                 # checkpoint. Better have one.

#### Mini Batches  #####
batch_size=5000                           # how many pictures per mini-batch. This number is huge! Use 50 instead
n_batches=int(np.floor(X_train.shape[0]/batch_size))   #how many minibatches

with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # If we have run this before, just continue 
        print("Parameters Found. Continuing")
        saver.restore(sess,save_file)    
    else:
        print("New Optimization Started")
        init.run()                       # If it is the first time, initialize.
        
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch=X_train[(batch_size*j):(batch_size*(j+1))]        #assign elements to X in first mini-batch
            y_batch=y_train[(batch_size*j):(batch_size*(j+1))]        #assign elements to X in first mini-batch
            sess.run(training_op,feed_dict={X: X_batch, y: y_batch})  # this is a training session and how to train it (training_op is defined above)
        if i % 10==0:
            print("Epoch:",i,"Loss:",loss.eval(feed_dict={X: X_batch, y: y_batch}),"accuracy:",accuracy.eval(feed_dict={X: X_batch, y: y_batch}))  # notice the eval()!!
        
    print("Final Loss:",loss.eval(feed_dict={X: X_batch, y: y_batch}),"Final Accuracy",accuracy.eval(feed_dict={X: X_batch, y: y_batch}))                 # Best result
    save_path = saver.save(sess, save_file)     # Save stuff for later. It returns a path ... 
    print("Model data saved in %s" % save_path)      # ... and we can well print it!
    
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    

from skimage.transform import rotate

train_set=pd.read_csv("../input/train.csv")  # read the CSV file

##### Prepare data with Pandas ######
X_train=train_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_train=X_train.values                          # only the numbers, no column labels 
y_train=train_set["label"]                      # Keep only column "label"
y_train=y_train.values                          # only the numbers, no column labels 

##### Augment data with skimage ######

# First we create two empty matrices with the same shape as X_train that contains the rotated images
X_rot20=np.zeros((X_train.shape[0],X_train.shape[1]))     
X_rotm20=np.zeros((X_train.shape[0],X_train.shape[1]))    
# then we rotate +20 and -20 degrees
for i in range (X_train.shape[0]):
    X_rot20[i]=rotate(X_train[i].reshape(28,28),20,preserve_range=True).reshape(1,784)
    X_rotm20[i]=rotate(X_train[i].reshape(28,28),-20,preserve_range=True).reshape(1,784)

##### New Data #####
X_train=np.concatenate((X_train,X_rot20,X_rotm20))    # Let create a new vector with all the data
y_train=np.concatenate((y_train,y_train,y_train))   # Even if we rotate an image, its answer does not change!

# Shuffle the indexes
shuffled_positions=np.random.permutation(X_train.shape[0])
X_train=X_train[shuffled_positions]
y_train=y_train[shuffled_positions]

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import  glob
from skimage.transform import rotate
#### Custom Functions to  move picture 1 pixel in every direction#####
def MoveUp(OneLinePic):
    tmp=OneLinePic.reshape(28,28)
    newPic=np.zeros((28,28))
    for i in range(27):
        newPic[i]=tmp[i+1]
    return newPic.reshape(1,784)

def MoveDown(OneLinePic):
    tmp=OneLinePic.reshape(28,28)
    newPic=np.zeros((28,28))
    for i in range(27):
        newPic[i+1]=tmp[i]
    return newPic.reshape(1,784)

def MoveLeft(OneLinePic):
    tmp=OneLinePic.reshape(28,28)
    newPic=np.zeros((28,28))
    for i in range(27):
        newPic[:,i]=tmp[:,i+1]
    return newPic.reshape(1,784)

def MoveRight(OneLinePic):
    tmp=OneLinePic.reshape(28,28)
    newPic=np.zeros((28,28))
    for i in range(27):
        newPic[:,i+1]=tmp[:,i]
    return newPic.reshape(1,784)

#### Here starts the code ####
tf.reset_default_graph()                     # to avoid error message in the notebook. Probably you will never use it in a real script
train_set=pd.read_csv("../input/train.csv")  # read the CSV file

##### Prepare data with Pandas ######
X_train=train_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_train=X_train.values                          # only the numbers, no column labels 
y_train=train_set["label"]                      # Keep only column "label"
y_train=y_train.values                          # only the numbers, no column labels 

##### Augment data with skimage ######

# First we create two empty matrices with the same shape as X_train that contains the rotated images
X_rot20=np.zeros((X_train.shape[0],X_train.shape[1]))     
X_rotm20=np.zeros((X_train.shape[0],X_train.shape[1]))    
# then we rotate +20 and -20 degrees
for i in range (X_train.shape[0]):
    X_rot20[i]=rotate(X_train[i].reshape(28,28),20,preserve_range=True).reshape(1,784)
    X_rotm20[i]=rotate(X_train[i].reshape(28,28),-20,preserve_range=True).reshape(1,784)
    
##### Augment data with Move Functions ######
X_up=np.zeros((X_train.shape[0],X_train.shape[1]))     
X_down=np.zeros((X_train.shape[0],X_train.shape[1]))    
X_left=np.zeros((X_train.shape[0],X_train.shape[1]))     
X_right=np.zeros((X_train.shape[0],X_train.shape[1]))    
for i in range (X_train.shape[0]):
    X_up[i]=MoveUp(X_train[i])
    X_down[i]=MoveDown(X_train[i])
    X_left[i]=MoveLeft(X_train[i])
    X_right[i]=MoveRight(X_train[i])

##### New Data #####
X_train=np.concatenate((X_train,X_rot20,X_rotm20,X_up,X_down,X_left,X_right))    # Let create a new vector with all the data
y_train=np.concatenate((y_train,y_train,y_train,y_train,y_train,y_train,y_train))   # Even if we rotate an image, its answer does not change!

# Shuffle the indexes
shuffled_positions=np.random.permutation(X_train.shape[0])
X_train=X_train[shuffled_positions]
y_train=y_train[shuffled_positions]

#### Normalize Data with scikit ####
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

###### TensorFlow starts here  ######
# Define the variables
X=tf.placeholder(shape=(None,28*28),dtype=tf.float32,name="X")  # training set 
y=tf.placeholder(shape=(None),dtype=tf.int64,name="y")    # Correct output: Not prediction, but correct answer!

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")

##### Grading ######
# For human beings : Accuracy
with tf.name_scope("Accuracy"):
    Guess=tf.nn.in_top_k(output_layer,y,1)    # Pick the digit with highest probability
    accuracy=tf.reduce_mean(tf.cast(Guess, tf.float32))  # Count how many you got right and give %

# For the neuronal network: Loss Function
with tf.name_scope("LossFunction"):
    EntropyProbability=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer) # compute the cross entropy
    loss=tf.reduce_mean(EntropyProbability,name="Loss") # compute average cross entropy

with tf.name_scope("Revision"):
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    training_op=optimizer.minimize(loss)

##### Lets run the neuronal network!  #####

init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)

n_epochs=50                              # how many "training sessions"
save_file="./modelAugmented.ckpt"                 # checkpoint. Better have one.

#### Mini Batches  #####
batch_size=5000                           # how many pictures per mini-batch. This number is huge! Use 50 instead
n_batches=int(np.floor(X_train.shape[0]/batch_size))   #how many minibatches

with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # If we have run this before, just continue 
        print("Parameters Found. Continuing")
        saver.restore(sess,save_file)    
    else:
        print("New Optimization Started")
        init.run()                       # If it is the first time, initialize.
        
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch=X_train[(batch_size*j):(batch_size*(j+1))]        #assign elements to X in first mini-batch
            y_batch=y_train[(batch_size*j):(batch_size*(j+1))]        #assign elements to y in first mini-batch
            sess.run(training_op,feed_dict={X: X_batch, y: y_batch})  # this is a training session and how to train it (training_op is defined above)
        if i % 10==0:
            print("Epoch:",i,"Loss:",loss.eval(feed_dict={X: X_batch, y: y_batch}),"accuracy:",accuracy.eval(feed_dict={X: X_batch, y: y_batch}))  # notice the eval()!!
        
    print("Final Loss:",loss.eval(feed_dict={X: X_batch, y: y_batch}),"Final Accuracy",accuracy.eval(feed_dict={X: X_batch, y: y_batch}))                 # Best result
    save_path = saver.save(sess, save_file)     # Save stuff for later. It returns a path ... 
    print("Model data saved in %s" % save_path)      # ... and we can well print it!
    
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import glob

tf.reset_default_graph()                     # to avoid error message in the notebook. Probably you will never use it in a real script
train_set=pd.read_csv("../input/train.csv")  # read the CSV file

##### Prepare data with Pandas ######
X_train=train_set.drop(labels=["label"],axis=1) # all data except the columns called "label". 
X_train=X_train.values                          # only the numbers, no column labels 
y_train=train_set["label"]                      # Keep only column "label"
y_train=y_train.values                          # only the numbers, no column labels 

#### Normalize Data with scikit ####
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

###### TensorFlow starts here  ######
# Define the variables
X=tf.constant(X_train,dtype=tf.float32,name="X")  #input ..... Why float?!
y=tf.constant(y_train,dtype=tf.int64,name="y")    # Correct output: Not prediction, but correct answer!

# Build the Neuronal Network
with tf.name_scope("NeuronalNetwork"): 
    input_layer=tf.layers.dense(X,300, name="InputLayer",activation=tf.nn.relu)
    middle_layer=tf.layers.dense(input_layer,100, name="MiddleLayer",activation=tf.nn.relu)
    output_layer=tf.layers.dense(middle_layer,10, name="OutputLayer")

##### Grading ######
# For human beings : Accuracy
with tf.name_scope("Accuracy"):
    Guess=tf.nn.in_top_k(output_layer,y,1)    # Pick the digit with highest probability
    accuracy=tf.reduce_mean(tf.cast(Guess, tf.float32))  # Count how many you got right and give %

# For the neuronal network: Loss Function
with tf.name_scope("LossFunction"):
    EntropyProbability=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output_layer) # compute the cross entropy
    loss=tf.reduce_mean(EntropyProbability,name="Loss") # compute average cross entropy

with tf.name_scope("Revision"):
    optimizer=tf.train.GradientDescentOptimizer(0.01)
    training_op=optimizer.minimize(loss)

#### Enable TensorBoard #####
summary_data=tf.summary.scalar('Loss',loss)       #Data we want to track and plot with TensorBoard
file_writer=tf.summary.FileWriter("TensorLogs",tf.get_default_graph())

##### Lets run the neuronal network!  #####

init=tf.global_variables_initializer()   # Allocate and initialize all the variables
saver=tf.train.Saver()                   # Store the values of the w's (and other stuff)

n_epochs=50                              # how many "training sessions"
save_file="./modelNormalized.ckpt"                  # checkpoint. Better have one.
with tf.Session() as sess:    
    if glob.glob(save_file + "*"):        # If we have run this before, just continue 
        print("Parameters Found. Continuing")
        saver.restore(sess,save_file)    
    else:
        print("New Optimization Started")
        init.run()                       # If it is the first time, initialize.
        
    for i in range(n_epochs):
        sess.run(training_op)   # this is a training session and how to train it (training_op is defined above)
        
        if i % 10==0:
            print("Epoch:",i,"Loss:",loss.eval(),"accuracy:",accuracy.eval())  # notice the eval()!!
            file_writer.add_summary(summary_data.eval(),i)     #Plot the data into the logdir
    print("Final Loss:",loss.eval(),"Final Accuracy",accuracy.eval())                 # Best result
    save_path = saver.save(sess, save_file)     # Save stuff for later. It returns a path ... 
    print("Model data saved in %s" % save_path)      # ... and we can well print it!
    
print ("Calculation Completed")         # Just to make sure we reached the end  of the program    
file_writer.close()