import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import svm

import tensorflow as tf



import matplotlib.pyplot as plt, matplotlib.image as mpimg

import matplotlib.cm as cm

%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')

print("there are {} total images".format(labeled_images.shape))
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
img_num = labeled_images.shape[0]

def preprocess(data, labeled = True):

    '''vector data(784) into image(28*28)'''

    images = []

    if labeled:

        images = data.iloc[:,1:]/255

    else:

        images = data/255

    print(images.shape)

    width = height = np.ceil(np.sqrt(images.shape[1])).astype(np.uint8)

    images = np.reshape(np.array(images), (-1, width, height, 1))

    print(images.shape)

    labels = []

    if labeled:

        labels = data.iloc[:,:1]

        labels_count = np.unique(labels).shape[0]

        print("There are {} labels.".format(labels_count))

        #labels = dense_to_one_hot(labels, labels_count)

        #print(labels[230])

        #labels = tf.one_hot(labels, labels_count)

        #tf.Print(labels[230], [labels[230]])

        labels = encoder.fit_transform(labels)

        print(labels[230])

        #print(labels.shape)

    return images, labels


images, labels = preprocess(labeled_images, labeled = True)
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)
# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch

X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# correct answers will go here

Y_ = tf.placeholder(tf.float32, [None, 10])

# variable learning rate

lr = tf.placeholder(tf.float32)

# dropout probability

pkeep = tf.placeholder(tf.float32)



# three convolutional layers with their channel counts, and a

# fully connected layer (the last layer has 10 softmax neurons)

# try another value(24, 48, 64, 200)

K = 6  # first convolutional layer output depth 24

L = 12  # second convolutional layer output depth 48

M = 24  # third convolutional layer 64

N = 200  # fully connected layer 200



W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels

B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))

W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))

B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))

W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))

B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))



W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))

B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))

B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
# The model

stride = 1  # output is 28x28

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)

stride = 2  # output is 14x14

Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)

stride = 2  # output is 7x7

Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)



# reshape the output from the third convolution for the fully connected layer

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])



Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

YY4 = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(YY4, W5) + B5

Y = tf.nn.softmax(Ylogits)
# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images

# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability

# problems with log(0) which is NaN

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)

cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# training step, the learning rate is a placeholder

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

train_a = [] 

test_a = []

train_range = []

test_range=[]

batch_size = 100

# You can call this function in a loop to train the model, 100 images at a time

def training_step(i, update_test_data, update_train_data):



    # training on batches of 100 images with 100 labels

    batch_X, batch_Y = next_batch(batch_size)



    # learning rate decay

    max_learning_rate = 0.003

    min_learning_rate = 0.0001

    decay_speed = 2000.0

    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)



    # compute training values for visualisation

    if update_train_data:

        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, pkeep: 1})

        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

        #datavis.append_training_curves_data(i, a, c)

        #datavis.update_image1(im)

        #datavis.append_data_histograms(i, w, b)

        train_a.append(a)

        train_range.append(i)

    # compute test values for visualisation

    if update_test_data:

        a, c = sess.run([accuracy, cross_entropy], {X: test_images, Y_: test_labels, pkeep: 1.0})

        print(str(i) + ": ********* epoch " + str(i*100//train_images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

        #datavis.append_test_curves_data(i, a, c)

        #datavis.update_image2(im)

        test_a.append(a)

        test_range.append(i)

   # the backpropagation training step

    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})

 #   sess.run(update_ema, {X: batch_X, Y_: batch_Y, itr: i, pkeep: 1.0})



#datavis.animate(training_step, 10001, train_data_update_freq=20, test_data_update_freq=100)

epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



def next_batch(batch_size):

    """Return the next `batch_size` examples from this data set."""

    global train_images

    global train_labels

    global epochs_completed 

    global index_in_epoch

    

    start = index_in_epoch

    # Shuffle for the first epoch

    if epochs_completed == 0 and start == 0:

      perm0 = np.arange(num_examples)

      np.random.shuffle(perm0)

      train_images = train_images[perm0]

      train_labels = train_labels[perm0]

    # Go to the next epoch

    if start + batch_size > num_examples:

      # Finished epoch

      epochs_completed += 1

      # Get the rest examples in this epoch

      rest_num_examples = num_examples - start

      images_rest_part = train_images[start:num_examples]

      labels_rest_part = train_labels[start:num_examples]

      # Shuffle the data

      

      perm = np.arange(num_examples)

      np.random.shuffle(perm)

      train_images = train_images[perm]

      train_labels = train_labels[perm]

      # Start next epoch

      start = 0

      index_in_epoch = batch_size - rest_num_examples

      end = index_in_epoch

      images_new_part = train_images[start:end]

      labels_new_part = train_labels[start:end]

      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)

    else:

      index_in_epoch += batch_size

      end = index_in_epoch

      return train_images[start:end], train_labels[start:end]
# init

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)



# to save the animation as a movie, add save_movie=True as an argument to datavis.animate

# to disable the visualisation use the following line instead of the datavis.animate line

for i in range(2000+1): 

    training_step(i, i % 100 == 0, i % 20 == 0)



#print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
plt.plot(train_range, train_a, '-b', label= "training")

plt.plot(test_range, test_a, '-g', label= "test")

plt.legend()

plt.xlabel('steps')

plt.ylabel('accuracy')
test_ims = pd.read_csv('../input/test.csv') 

print('there are {} images for test.'.format(test_ims.shape))

images, _ = preprocess(test_ims, labeled = False)

predict = tf.argmax(Y, 1)

test_num = test_ims.shape[0]

predicted_labels = np.zeros(test_num)

batch_size = 100

for i in range(test_num//batch_size):

    start = i*batch_size

    end = (i+1)*batch_size

    predicted_labels[start:end] = sess.run(predict, {X: images[start:end], pkeep: 0.75})

print(predicted_labels[:20])

im_num = 26

plt.figure(figsize = (20,16))

for i in range(6):

    plt.subplot(2, 3, i+1)

    plt.imshow(images[im_num+i*10, :, :, 0])

    plt.title("Predicted as {}".format(predicted_labels[im_num+i*10]))
np.savetxt('submission_softmax.csv',

          np.c_[range(1,len(test_ims)+1), predicted_labels],

          delimiter = '',

          header = 'ImageId, Label',

          comments = '',

          fmt = '%d')
sess.close()