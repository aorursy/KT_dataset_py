import numpy as np
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input/Sign-language-digits-dataset"]).decode("utf8"))
x = np.load("../input/Sign-language-digits-dataset/X.npy")
y = np.load("../input/Sign-language-digits-dataset/Y.npy")

x = x.reshape(x.shape[0], 64, 64, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x.shape, x.dtype)
print(y.shape)
# Hyperparameters
num_filters = [64,128,256,256]
strides = [1,2]
image_height, image_width, image_channels = 64,64,1
learning_rate = 1e-3
keep_prob = 0.7
num_hidden = 128
num_epochs = 300
log_every = 10
test_every = 10
save_every = 50
batch_size = 64
def rprint(string):
    sys.stdout.write("\r{}".format(string))
    sys.stdout.flush()
def conv_layer(name, x, filter_size, input_channels, output_channels, strides=(1, 1), padding='SAME'):
    with tf.variable_scope(name):
        filters = tf.get_variable(name="filters", initializer= tf.random_normal_initializer, shape=[filter_size[0], filter_size[1], input_channels, output_channels], dtype=tf.float32, )
        bias = tf.get_variable(name="bias", initializer= tf.random_normal_initializer, shape=[output_channels],dtype=tf.float32)
        conv2d_op = tf.nn.conv2d(input=x, filter=filters, strides=[1, strides[0], strides[1], 1], padding=padding)
        conv2d_op = tf.nn.bias_add(conv2d_op, bias)
        return conv2d_op
def max_pool_layer(x, kernel_shape, strides=(2,2), padding="SAME"):
    max_pool_op = tf.nn.max_pool(value=x,
                                 ksize=[1, kernel_shape[0], kernel_shape[1], 1], 
                                 strides=[1,strides[0], strides[1], 1], 
                                 padding=padding, 
                                 name="maxpool"
                                )
    return max_pool_op
def leaky_relu(x, leakiness=0.0):
    return tf.nn.leaky_relu(features=x, alpha=leakiness, name="leaky_relu")
def build_model(x,dropout_keepprob):
    with tf.variable_scope("CNN"):
        with tf.variable_scope("layer-1"):
            x = conv_layer("cnn-1",x,(3,3),image_channels,num_filters[0])
            x = leaky_relu(x,0.01)
            x = max_pool_layer(x,(2,2))
        print("layer-1",x.get_shape().as_list())
        
        with tf.variable_scope("layer-2"):
            x = conv_layer("cnn-2",x,(3,3),num_filters[0],num_filters[1])
            x = leaky_relu(x,0.01)
            x = max_pool_layer(x,(2,2))
        print("layer-2",x.get_shape().as_list())
        
        with tf.variable_scope("layer-3"):
            x = conv_layer("cnn-3",x,(3,3),num_filters[1],num_filters[2])
            x = leaky_relu(x,0.01)
            x = max_pool_layer(x,(2,2))
        print("layer-3",x.get_shape().as_list())
        
        with tf.variable_scope("layer-4"):
            x = conv_layer("cnn-4",x,(3,3),num_filters[2],num_filters[3])
            x = leaky_relu(x,0.01)
            x = max_pool_layer(x,(2,2))
        print("layer-4",x.get_shape().as_list())
    
    x_shape = x.get_shape().as_list()[1:]
    with tf.variable_scope("FC"):
        with tf.variable_scope("layer-1"):
            weights = tf.get_variable(name="weights-1", initializer=tf.random_normal_initializer, shape=[np.prod(x_shape),2048], dtype=tf.float32)
            biases = tf.get_variable(name="biases-1", initializer=tf.random_normal_initializer, shape=[2048], dtype=tf.float32 )
            
            x = tf.reshape(x,[-1,np.prod(x_shape)])
            x = leaky_relu(tf.matmul(x,weights)+biases)
            x = tf.nn.dropout(x,dropout_keepprob)
        
        with tf.variable_scope("layer-2"):
            weights = tf.get_variable(name="weights-2", initializer=tf.random_normal_initializer, shape=[2048,1024], dtype=tf.float32)
            biases = tf.get_variable(name="biases-2", initializer=tf.random_normal_initializer, shape=[1024], dtype=tf.float32)
            
#             x = tf.reshape(x,[-1,np.prod(x_shape)])
            x = leaky_relu(tf.matmul(x,weights)+biases)
            x = tf.nn.dropout(x,dropout_keepprob)
        
        with tf.variable_scope("layer-3"):
            weights = tf.get_variable(name="weights-3", initializer=tf.random_normal_initializer, shape=[1024,10], dtype=tf.float32)
            biases = tf.get_variable(name="biases-3", initializer=tf.random_normal_initializer, shape=[10], dtype=tf.float32)
            
            x = tf.matmul(x,weights)+biases
            
        
    return x
def build_output_ops(logits,labels):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return [train_op, loss_op, accuracy]
X = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channels], name="inputs")
Y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
dropout_keepprob = tf.placeholder_with_default(1.0,shape=(), name="dropout_keepprob")

logits = build_model(X,dropout_keepprob)
train_ops = build_output_ops(logits,Y)
best_accuracy = -1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1,num_epochs+1):
        for batch_index in range(0,len(x_train),batch_size):
            feed_dict = {X:x_train[batch_index:batch_index+batch_size], Y:y_train[batch_index:batch_index+batch_size], dropout_keepprob:keep_prob}
            _, loss_val, accuracy_val = sess.run(train_ops, feed_dict=feed_dict)
            rprint("Epoch: {}, Batch: {:0>4d}/{}, Loss: {:.2f}, Accuracy: {:.2%}".format(epoch,batch_index,len(x),loss_val,accuracy_val))
        if epoch%test_every==0:
            accuracies = []
            for batch_index in range(0,len(x_test),batch_size):
                feed_dict = {X:x_test[batch_index:batch_index+batch_size], Y:y_test[batch_index:batch_index+batch_size]}
                accuracy_val = sess.run(train_ops[2], feed_dict=feed_dict)
                accuracies.append(accuracy_val)
            test_accuracy = np.mean(accuracies)
            if test_accuracy>best_accuracy:
                best_accuracy=test_accuracy
            print()
            print("Test Accuracy after {} epochs: {:.2%}".format(epoch,test_accuracy))
print("Best accuracy on test set:",best_accuracy)
