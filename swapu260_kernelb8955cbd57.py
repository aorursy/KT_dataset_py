# all necessary imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input/Sign-language-digits-dataset"]).decode("utf8"))
# load the training data into numpy array

from sklearn.model_selection import train_test_split

X = np.load('../input/Sign-language-digits-dataset/X.npy')
Y = np.load('../input/Sign-language-digits-dataset/Y.npy')

test_size = 0.2

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# looking at the shape and size of training and testing data 
print("Training data")
print("\tFeatures")
print("\tNo. of examples in training data: {}".format(X_train.shape[0]))
print("\tHeight and width of images in training data (shape of data): {}, {}\n".format(X_train.shape[1],X_train.shape[2]))

print("\tlabels")
print("\tNumber of labels {}:".format(Y_train.shape[0]))
print("\tShape of labels {}:\n".format(Y_train.shape[1:3]))


print("Testing data")
print("\tlabels")
print("\tNo. of examples in training data: {}".format(X_test.shape[0]))
print("\tHeight and width of images in training data (shape of data): {}, {}\n".format(X_test.shape[1],X_test.shape[2]))

print("\tlabels")
print("\tNumber of labels {}:".format(Y_test.shape[0]))
print("\tShape of labels {}:".format(Y_test.shape[1:3]))
# A look at data gives better understanding about the data 
# you can enter any number between 0 and 1649 as there are 1649 images
plt.imshow(X_train[100])
print("label for X_train[{}]: {}".format(100, Y_train[100]))
# display some training examples 
img_size = X.shape[1]

print("Examples")
print("Corresponding labels:")
n = 10
plt.figure(figsize=(20,4))
for i in range(1, n+1):
    p = plt.subplot(1, n, i)
    plt.imshow(X_train[i].reshape([img_size, img_size]))
    print("{}".format(np.argmax(Y_train[i])), end="\t")
    plt.gray()
    plt.axis("off")
tf.logging.set_verbosity(tf.logging.INFO)
# using tensorflow estimator api to create and test model 
# going to use the same model as used for mnist data 

def model(features, labels, mode):
    """ Model function for convolutional neural nets.
        Args: 
            features: training examples with shape (Mx64x64)
            labels: respective labels for the training example
            mode: mode of operation i.e train, evaluate, predict
    """
    # Input layer 
    # -1 for batch size which will be dynammically computed, height-64, width-64, channels-1
    input_layer = tf.reshape(features, [-1, 64, 64, 1]) 
    
    # convolution layer
    # after this layer shape of volume will be (Mx64x64x32)
    # where 32 is number of filters
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters=32, 
        kernel_size= [5,5],
        padding='same',
        activation=tf.nn.relu)
    
    # pooling layer 1 
    # after pooling shape of volume will be (64-2)/2 + 1 = 32
    # (Mx32x32x32)
    pool1  = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2)
    
    # convolution layer #2 and Pooling layer #2
    # after 2nd convolution shape will be (Mx32x32x64)
    conv2 = tf.layers.conv2d(
        inputs= pool1,
        filters=64, 
        kernel_size= [5,5],
        padding='same',
        activation=tf.nn.relu)
    
    # after 2nd pooling shape will be (Mx16x16x64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # dense layer 
    # fully connected layer
    # converted 16x16x32 matrix to (1, 16*168*64)
    pool2_flat = tf.reshape(pool2, [-1, 16*16*64])
    
    dense_layer = tf.layers.dense(inputs= pool2_flat, units=1024, activation= tf.nn.relu)
    # dropout layer to dropout values below threshold
    dropout = tf.layers.dropout(
        inputs=dense_layer, rate=0.4, training=mode ==tf.estimator.ModeKeys.TRAIN)
    
    # logit layer or output layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    # for prediction 
    predictions = {
        # generate predictoins for predict and eval mode 
        'classes': tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        'probabilites': tf.nn.softmax(logits, name="softmax_tensor") 
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)
    
    # calculate loss for both training and eval mode 
    # cross entropy loss 
    print(labels.shape)
    print(logits.shape)
    loss = tf.losses.softmax_cross_entropy(labels, logits=logits)
    
    # configure for training operation 
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)
    
    # configure for evaluation metrices 
    eval_metric_ops = {
        "accuracy": tf.metrices.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
# driver program 
def main(unused_argv):
    
    # create the estimator 
    classifier = tf.estimator.Estimator(
        model_fn=model)
    # setting up logging for predictions 
    tensors_to_log = {'probabilities': "softmax_tensor" }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter= 100)
    
    # train the model 
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X_train,
        y=Y_train,
        batch_size=32,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])
    
    # Evaluate the model 
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X_test,
        y=Y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    

    
    
# run application 
if __name__=="__main__":
    tf.app.run()
