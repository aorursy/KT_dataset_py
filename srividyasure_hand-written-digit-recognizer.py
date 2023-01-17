import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
feature_data_train = train_data.iloc[:,1:]
target_data_train  = train_data.iloc[:,0]
feature_data_test = test_data
feature_data_train.head()
#target_data_train = pd.get_dummies(target_data_train)
target_data_train.head()
trainX, testX, trainY,testY = train_test_split(feature_data_train,target_data_train, test_size = 0.2, random_state = 1212)
testX, validationX, testY, validationY = train_test_split(testX,testY,test_size=0.5,random_state=0)
print("taining data set shape feature data:",trainX.shape)
print("taining data set shape label or target data:",trainY.shape)
print("validation data set shape feature data:", validationX.shape)
print("validation data set shape label or target data:",validationY.shape)
print("testing data set shape feature data:", testX.shape)
print("testing data set shape label or target data:",testY.shape)
def data_preparation(dataset, labels):
    image_pixel_row = 28
    image_pixel_col = 28
    image_pixel_channel = 1 #gray scale 1 #RGB 3
    dataset = dataset.as_matrix()
    labels = labels.as_matrix()
    dataset = dataset.reshape((-1,image_pixel_row,image_pixel_col,image_pixel_channel)).astype(np.float32)
    labels =(np.arange(10) == labels[:,None]).astype(np.float32)
    
    return dataset, labels
   
trainX , trainY = data_preparation(trainX,trainY)
validationX,validationY = data_preparation(validationX,validationY)
testX,testY = data_preparation(testX,testY)
print("taining data set shape :",trainX.shape,"|",trainY.shape)
print("validation data set shape:", validationX.shape ,"|",validationY.shape)
print("testing data set shape:", testX.shape ,"|",testY.shape)
# Pad images with 0s
trainX = np.pad(trainX, ((0,0),(2,2),(2,2),(0,0)), 'constant')
validationX = np.pad(validationX, ((0,0),(2,2),(2,2),(0,0)), 'constant')
testX = np.pad(testX, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print("taining data set shape :",trainX.shape,"|",trainY.shape)
print("testing data set shape:", validationX.shape ,"|",validationY.shape)
print("testing data set shape:", testX.shape ,"|",testY.shape)
def model(feature_data):
    # Edge Detector Convolution Input = 32x32x1, Output = 32x32x1
    edge_kernel_1 =np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])
    
    conv_w = tf.constant(edge_kernel_1, dtype=tf.float32, shape=(3, 3, 1, 1))
    transf_edge1 = tf.nn.conv2d(feature_data,conv_w,strides = [1,1,1,1], padding = 'SAME')
    
    
    # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
    conv_layer1_weights = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
    conv_layer1_baises = tf.Variable(tf.zeros(6))
    conv_layer1 = tf.add(tf.nn.conv2d(transf_edge1,conv_layer1_weights,strides=[1,1,1,1],padding='VALID'),conv_layer1_baises)
    
    #Activation
    conv_layer1 = tf.nn.leaky_relu(conv_layer1)
    
    # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
    layer_1_out = tf.nn.avg_pool(conv_layer1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
    
    # Layer 2: Convolutional.Input = 14x14x6 Output = 10x10x16.
    conv_layer2_weights = tf.Variable(tf.truncated_normal(shape = [5,5,6,16],mean = 0, stddev = 0.1))
    conv_layer2_baises = tf.Variable(tf.zeros(16))
    conv_layer2 = tf.add(tf.nn.conv2d(layer_1_out,conv_layer2_weights,strides=[1,1,1,1],padding='VALID'),conv_layer2_baises)
    
    # Activation.
    conv_layer2 = tf.nn.leaky_relu(conv_layer2)
    
  
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer_2_out = tf.nn.avg_pool(conv_layer2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        
    # Layer 3 input = 5x5x16 output = 3x3x32
    conv_layer_new_weights = tf.Variable(tf.truncated_normal(shape=[3,3,16,32],mean=0,stddev = 0.1))
    conv_layer_new_baises = tf.Variable(tf.zeros(32))
    conv_layer_new = tf.add(tf.nn.conv2d(layer_2_out,conv_layer_new_weights,strides=[1,1,1,1],padding='VALID'),conv_layer_new_baises)
    
    # Activation.
    conv_layer_new = tf.nn.leaky_relu(conv_layer_new)
    
    # Pooling. Input = 3x3x32. Output = 1x1x32.
    conv_layer_new = tf.nn.avg_pool(conv_layer_new, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    
    # Flatten. Input = 1x1x32. Output = 32.
    flatten_layer_1_input = flatten(conv_layer_new)
    
    # Layer 4: Fully Connected. Input = 32. Output = 25.
    flatten_layer_1_weights = tf.Variable(tf.truncated_normal(shape = (32,25),mean=0,stddev=0.1))
    flatten_layer_1_biases = tf.Variable(tf.zeros(25))
    flatten_layer_1 = tf.add(tf.matmul(flatten_layer_1_input,flatten_layer_1_weights), flatten_layer_1_biases)
    
    # Activation.
    flatten_layer_1_out = tf.nn.leaky_relu(flatten_layer_1)
    
    # Layer 5: Fully Connected. Input = 25. Output = 16.
    flatten_layer_2_weights = tf.Variable(tf.truncated_normal(shape = (25,16),mean=0,stddev=0.1))
    flatten_layer_2_biases = tf.Variable(tf.zeros(16))
    flatten_layer_2 = tf.add(tf.matmul(flatten_layer_1_out,flatten_layer_2_weights), flatten_layer_2_biases)
    
    # Activation.
    flatten_layer_2_out = tf.nn.leaky_relu(flatten_layer_2)
    
    # Layer 6: Fully Connected. Input = 16. Output = 10.
    flatten_layer_3_weights = tf.Variable(tf.truncated_normal(shape = (16,10),mean=0,stddev=0.1))
    flatten_layer_3_biases = tf.Variable(tf.zeros(10))
    output = tf.add(tf.matmul(flatten_layer_2_out,flatten_layer_3_weights), flatten_layer_3_biases)
    
    return output
X = tf.placeholder(tf.float32, shape=[None,32,32,1])
Y_ = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(Y_, 10)
predictions = model(X)
#Softmax with cost function implementation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=Y_)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)
# Evaluation
correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(Y_,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def evaluate_model(feature_data, labels):
    number_of_examples = len(feature_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, number_of_examples, BATCH_SIZE):
        batch_X, batch_Y = feature_data[offset:offset+BATCH_SIZE], labels[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,feed_dict={X: batch_X, Y_: batch_Y})
        total_accuracy += (accuracy*len(batch_X))
    
    return total_accuracy / number_of_examples
        

EPOCHS = 100
BATCH_SIZE = 128

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    number_of_exmaples = len(trainX)
    print("Training... with dataset - ", number_of_exmaples)
    print()
    
    for i in range(EPOCHS):
        trainX, trainY = shuffle(trainX,trainY)
        for offset in range(0,number_of_exmaples,BATCH_SIZE):
            end = offset+BATCH_SIZE
            batchX,batchY = trainX[offset:end],trainY[offset:end]
            sess.run(training_operation,feed_dict={X: batchX, Y_: batchY})
        validation_accuracy = evaluate_model(validationX, validationY)
        print("EPOCH",i+1)
        print("Validation Accuracy:",validation_accuracy)
        print()
    saver = tf.train.Saver()
    save_path = saver.save(sess, '/tmp/lenet.ckpt')
    print("Model saved %s "%save_path)
    
    test_accuracy = evaluate_model(testX, testY)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
feature_data_test = feature_data_test.as_matrix()
feature_data_test = feature_data_test.reshape((-1,28,28,1)).astype(np.float32)
feature_data_test = np.pad(feature_data_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/lenet.ckpt")
    print("Model restored.")
    Z = predictions.eval(feed_dict={X: feature_data_test})
    y_pred = np.argmax(Z, axis=1)
    print(len(y_pred))
 
    #Write into a CSV file with columns ImageId & Label 
    submission = pd.DataFrame({
    "ImageId": list(range(1, len(y_pred)+1)),
    "Label": y_pred
 })
    submission.set_index(['ImageId'],inplace=True)
    submission.to_csv("mnist.csv")