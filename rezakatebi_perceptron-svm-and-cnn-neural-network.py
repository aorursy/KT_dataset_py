import numpy as np

import timeit

import pandas as pd

from sklearn import svm

import tensorflow as tf 

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline



sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../input/fashion',one_hot = True, validation_size=0)

#y = data.train.labels

#print(y.shape)
fig, axes = plt.subplots(3,4, figsize=(10,10))

for row in axes:

    for ax in row:

        batch_X, batch_Y = data.train.next_batch(100)

        image = batch_X[0].reshape(28,28)

        ax.imshow(image, cmap = 'gray')
data_New = input_data.read_data_sets('../input/fashion',one_hot = False)

# Training dataset 

dataset,labels = data_New.train.images, data_New.train.labels



# Testing dataset 

data_tes, labels_tes = data_New.test.images, data_New.test.labels





#########################################################

#########################################################

#########################################################

# DEFINING FUNCTIONS THAT WE WILL NEED 

# This function labeles 1 if the label is equal to 

# digit and -1 otherwise

def digits_label(X):

    digits = np.arange(0,10)

    labels = np.zeros((X.shape[0], digits.shape[0]))

    for i in digits:

        for j in range(X.shape[0]):

            if X[j] == i:

                labels[j,i] = 1

            else:

                labels[j,i] = -1

    return labels



# This function will be used to show the confusion matrix 

def confusion_mat(x,y,n_classes):

    mat = np.zeros((n_classes, n_classes))

    for a,b in zip(x,y):

        mat[int(a)][int(b)] += 1

    return mat



# This a sign fuction 

def sign_(X):

    if X >= 0 :

        return 1.0

    else:

        return -1.0

    

###############################################

###############################################    

# Input:

#    'data' is a 2D array, with one exampel per row.

#    'labels' is a 1D array of labels for the corresponding examples.

#     'epochs' is the maximum number of epochs.

# Output:

#     the weight vector w.

def perceptron_train(data, labels, epochs):

    w = np.zeros(data.shape[1])

    for j in range(epochs):

        for i in range (data.shape[0]):

            h = sign_(data[i,:].dot(w))

            if (h!=labels[i]):

                w += labels[i] * data[i,:]

    return w



###################################################

####################################################



# Input:

#    'data' is a 2D array, with one exampel per row.

#    'labels' is a 1D array of labels for the corresponding examples.

#     'epochs' is the maximum number of epochs.

#     'kernel' is the kernel function to be used.

# Output:

#     the parameter vector alpha.

def kperceptron_train(data, labels, epochs, kernel,p):

    n_examples, n_features = data.shape

    alpha = np.zeros(n_examples, dtype=np.float64)



    K = np.zeros((n_examples, n_examples))

    for i in range(n_examples):

        for j in range(n_examples):

            K[i,j] = kernel(data[i], data[j],p)

            

    for t in range(epochs):

        for i in range(n_examples):

            if sign_(np.sum(K[:,i] * alpha)) != labels[i]:

                alpha[i] += labels[i]

        

    return alpha



####################################################

####################################################



# Input:

#    'alpha' is the parameter vector.

#    'data' is a 2D array, with one exampel per row.

#     'kernel' is the kernel function to be used.

# Output:

#     a vector with the predicted labels for the examples in 'data'.



def kperceptron_pred(alpha, data_tra, data_tes, kernel,p):

    n_examples_tra = data_tra.shape[0]

    n_examples_tes  = data_tes.shape[0]

    prediction = np.zeros(n_examples_tes, dtype=np.float64)



    K = np.zeros((n_examples_tra, n_examples_tes))

    for i in range(n_examples_tra):

        for j in range(n_examples_tes):

            K[i,j] = kernel(data_tra[i], data_tes[j],p)



    for i in range(n_examples_tes):

        prediction[i] = np.sum(K[:,i] * alpha)

    return prediction





####################################################

####################################################



# Input: two examples x and y and p .

# Output: the kernel value computed as (1+xTy)^p.

def poly_kernel(x, y, p):

    return (1 + np.dot(x, y)) ** p



def gauss_kernel(x,y,p):

    return np.exp(-np.dot(x-y,x-y)/(2*p**2))



######################################################

######################################################

# 1 if the label equal to the digit and -1 otherwise

labels_tf = digits_label(labels)



# Train and find which epoch gives highest accuracy

start = timeit.default_timer()

Epochs = np.arange(1,10,1)

digits = np.arange(0,10)

accs = []

w = []

for T in Epochs:

    preds =[]

    w_digits = []

    for i in digits:

        w_t = perceptron_train(dataset[5500:,:], labels_tf[5500:,i], T)

        w_t = w_t / np.linalg.norm(w_t)

        pred = dataset[:5500,:].dot(w_t)

        preds.append(pred)

        w_digits.append(w_t)

    w.append(w_digits)

    labels_pred = np.argmax(preds, axis = 0)

    acc = np.mean(labels[:5500] == labels_pred) * 100

    accs.append(acc)

w = np.asarray(w, dtype=np.float64)

stop = timeit.default_timer()

#np.savetxt(FLAGS.input_data_dir + 'W_lin_per_best_T.txt',w[np.argmax(accs)])

print('##################################################')

print('Testing linear perceptron on development data set.')

print('The highest accuracy of %s was achieved after %s epochs.'%(np.max(accs), Epochs[np.argmax(accs)]))

print('The training time for linear perceptron was %s seconds'%(stop - start))



##################################################

##################################################

# Testing on the test data set

preds  = []

for l in w[np.argmax(accs)]:

    pred = data_tes.dot(l)

    preds.append(pred)

labels_pred = np.argmax(preds, axis = 0)

acc_tes = np.mean(labels_tes == labels_pred) * 100

print("Accuracy on the test dataset is :", acc_tes)



conf = confusion_mat(labels_pred, labels_tes, 10)



# This part plots confusion matrix 

df_conf = pd.DataFrame(conf, range(10),range(10))

sn.set(font_scale=1.4)#for label size

plt.title('Confusion matrix \nLinear perceptron')

conf_plot = sn.heatmap(df_conf, annot=True,annot_kws={"size": 7})# font size

conf_plot.set(xlabel='True', ylabel='Predicted')

plt.figure(figsize=(60,60))

plt.show()

plt.close()
'''

start = timeit.default_timer()

# Number of epochs tuned uisng Linear perceptron

T = 7

d = np.arange(2,7)

digits = np.arange(0,10)

accs = []

alpha = []

for p in d:

    preds =[]

    alpha_digits = []

    for i in digits:

        alpha_t = kperceptron_train(dataset[100:1000,:], labels_tf[100:1000,i], T, kernel = poly_kernel, p=p)

        pred = kperceptron_pred(alpha_t, dataset[100:1000,:],dataset[:100,:], kernel = poly_kernel, p=p)

        preds.append(pred)

        alpha_digits.append(alpha_t)

        print(i)

    alpha.append(alpha_digits)

    labels_pred = np.argmax(preds, axis = 0)

    acc = np.mean(labels[:100] == labels_pred) * 100

    print(acc)

    accs.append(acc)

alpha = np.asarray(alpha, dtype=np.float64)

stop = timeit.default_timer()

#np.savetxt(FLAGS.input_data_dir + 'Alpha_ker_per_best_d.txt',alpha[np.argmax(accs)])

print('##################################################')

print('Testing polynomiyal kernel perceptron on development data set.')

print('The highest accuracy of %s was achieved for %s degree polynimiyal kernel.'%(np.max(accs), np.argmax(accs) + 2))

print('The training time for polynomiyal kernel perceptron was %s seconds'%(stop - start))





#################################################################################

# Test kernel perceptron on the test data set



#alpha_best = np.loadtxt(FLAGS.input_data_dir + 'Alpha_ker_per_best_d.txt')

preds  = []

for l in alpha[np.argmax(accs)]:

    pred = kperceptron_pred(l, dataset[100:1000,:],data_tes[:100], kernel = poly_kernel, p=2)

    preds.append(pred)

labels_pred = np.argmax(preds, axis = 0)

acc_tes = np.mean(labels_tes[:100] == labels_pred) * 100

print("Accuracy on the test dataset is :", acc_tes)



conf = confusion_mat(labels_pred[:100], labels_tes, 10)



# This part plots confusion matrix 

df_conf = pd.DataFrame(conf, range(10),range(10))

sn.set(font_scale=1.4)#for label size

plt.title('Confusion matrix \nPolynimiyal kernel perceptron d=2')

conf_plot = sn.heatmap(df_conf, annot=True,annot_kws={"size": 10})# font size

conf_plot.set(xlabel='True', ylabel='Predicted')

plt.show()

plt.close()

'''
# Train gaussian kernel Perceptron for each sigma and for each digit and test on development data

'''

start = timeit.default_timer()

T = 20

sig = np.array([0.1,2,10])

digits = np.arange(0,10)

accs = []

alpha = []

for p in sig:

    preds =[]

    alpha_digits = []

    for i in digits:

        alpha_t = kperceptron_train(dataset[5500:,:], labels_tf[5500:,i], T, kernel = gauss_kernel, p=p)

        pred = kperceptron_pred(alpha_t, dataset[5500:,:],dataset[:5500,:], kernel = gauss_kernel, p=p)

        preds.append(pred)

        alpha_digits.append(alpha_t)

    alpha.append(alpha_digits)

    labels_pred = np.argmax(preds, axis = 0)

    acc = np.mean(labels[:5500] == labels_pred) * 100

    print(acc)

    accs.append(acc)

alpha = np.asarray(alpha, dtype=np.float64)

#np.savetxt(FLAGS.input_data_dir + 'Alpha_ker_per_best_gauss.txt',alpha[np.argmax(accs)])

stop = timeit.default_timer()

print('##################################################')

print('Testing gauss kernel perceptron on development data set.')

print('The highest accuracy of %s was achieved for sigma = %s  in gaussian kernel.'%(np.max(accs), sig[np.argmax(accs)]))

print('The training time for gaussian kernel perceptron was %s seconds'%(stop - start))







#alpha_best = np.loadtxt(FLAGS.input_data_dir + 'Alpha_ker_per_best_gauss.txt')

preds  = []

for l in alpha[np.argmax(accs)]:

    pred = kperceptron_pred(l, dataset[5500:,:],data_tes, kernel = gauss_kernel, p=sig[np.argmax(accs)])

    preds.append(pred)

labels_pred = np.argmax(preds, axis = 0)

acc_tes = np.mean(labels_tes == labels_pred) * 100

print("Accuracy on the test dataset is :", acc_tes)



conf = confusion_mat(labels_pred, labels_tes, 10)



# This part plots confusion matrix 

df_conf = pd.DataFrame(conf, range(10),range(10))

sn.set(font_scale=1.4)#for label size

plt.title('Confusion matrix \nGauss kernel perceptron sigma=2')

conf_plot = sn.heatmap(df_conf, annot=True,annot_kws={"size": 10})# font size

conf_plot.set(xlabel='True', ylabel='Predicted')

plt.show()

plt.close()

'''
# Train and test poly kernel svm

'''

degrees = np.arange(2,7)

accs = []

start = timeit.default_timer()

for d in degrees:

    ss = svm.SVC(kernel = 'poly', degree = d, C = 1.0) 

    ss.fit(dataset[5500:,:],labels[5500:])

    pred = ss.predict(dataset[:5500,:])

    acc = np.mean(pred == labels[:5500])

    print(acc)

    accs.append(acc)

stop = timeit.default_timer()

print(accs)

print('The highest accuracy for svm is %s in degree equal to %s'%(np.max(accs), degrees[np.argmax(accs)]))

print('Time took for training svm poly kernel is %s seconds.'%(stop - start))   

ss = svm.SVC(kernel = 'poly', degree = degrees[np.argmax(accs)], C = 1.0) 

ss.fit(dataset[5500:,:],labels[5500:])

predict = ss.predict(data_tes)

acc = np.mean(predict == labels_tes) * 100

print('Accuracy on test dataset is %s.'% acc)

conf_poly = confusion_mat(predict, labels_tes, 10)



# This part plots confusion matrix 

df_conf_poly = pd.DataFrame(conf_poly, range(10),range(10))

sn.set(font_scale=1.4)#for label size

plt.title('Confusion matrix \nPolynimiyal kernel SVM d=5')

conf_plot = sn.heatmap(df_conf_poly, annot=True,annot_kws={"size": 10})# font size

conf_plot.set(xlabel='True', ylabel='Predicted')

plt.show()

plt.close()

'''
x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])



W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x,W) + b



sess.run(tf.global_variables_initializer())

cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))



train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)



start = timeit.default_timer()

for _ in range(20000):

    batch_x,batch_y = data.train.next_batch(100)

    train_step.run(feed_dict={x: batch_x, y_: batch_y})

stop = timeit.default_timer()

    

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_smax = accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels})



print('Accuracy of Softmax on test dataset is:',acc_smax * 100)

print('The training time for Softmax Regression was %s seconds'%(stop - start))
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)





def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                          strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = []

accuracy_range = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(30000):

        batch = data.train.next_batch(50)

        if i % 100 == 0:

            train_accuracy = accuracy.eval(feed_dict={

                x: batch[0], y_: batch[1], keep_prob: 1.0})

            print('step %d, training accuracy %g' % (i, train_accuracy))

            epochs = np.append(epochs,i)

            accuracy_range = np.append(accuracy_range,train_accuracy)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



    print('test accuracy %g' % accuracy.eval(feed_dict={

        x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
plt.plot(epochs,accuracy_range)

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("The accuracy vs number of epochs")