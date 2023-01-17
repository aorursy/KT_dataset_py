%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt

from IPython import display

import numpy as np

import pandas as pd

import tensorflow as tf
# parameters

LEARNING_RATE = 0.003

BATCH_SIZE = 100

ITERATION_COUNT = 10000

ACCURACY_CHECK = 100

TOTAL_COUNT = 42000 # training image count in input csv file

TRAIN_COUNT = 40000

TEST_COUNT = 2000
# reading train and test file

train = pd.read_csv("../input/train.csv")

pred  = pd.read_csv("../input/test.csv")
# extracting pixel values and label for each image

# splitting images to train and test data to validate our prediction

X_train  = train.iloc[:TRAIN_COUNT, 1:].values

X_test   = train.iloc[TRAIN_COUNT:, 1:].values

Y_values = train["label"].values

X_pred   = pred.values

X_all = train.iloc[TOTAL_COUNT:, 1:].values



# input pixel values range is [0:255]

# we need to change the range to [0:1]

# the easiest way is to devide by maximum 255

X_train = X_train / 255

X_test  = X_test / 255

X_pred  = X_pred / 255

X_all = X_all / 255



# creating list for each label

# each row's values will be zero 

# except the index of the label

# mapping should be as below: 

# 0 ~ [1 0 0 0 0 0 0 0 0 0]

# 1 ~ [0 1 0 1 0 0 0 0 0 0]

# 2 ~ [0 0 1 0 0 0 0 0 0 0]

# 3 ~ [0 0 0 1 0 0 0 0 0 0]

# 4 ~ [0 0 0 0 1 0 0 0 0 0]

# 5 ~ [0 0 0 0 0 1 0 0 0 0]

# 6 ~ [0 0 0 0 0 0 1 0 0 0]

# 7 ~ [0 0 0 0 0 0 0 1 0 0]

# 8 ~ [0 0 0 0 0 0 0 0 1 0]

# 9 ~ [0 0 0 0 0 0 0 0 0 1]

Y_labels = np.zeros(shape=(TOTAL_COUNT,10))



for i in range(TOTAL_COUNT):

    Y_labels[i][Y_values[i]] = 1



# splitting labels to train and test data to validate our prediction

Y_train = Y_labels[:TRAIN_COUNT]

Y_test  = Y_labels[TRAIN_COUNT:TOTAL_COUNT]

Y_all = Y_labels
def plot_batch_images(images, labels, predictions = None, count = 100):

    

    plt.rcParams['axes.linewidth'] = 0.5

    count_row_col = np.sqrt(count)



    fig = plt.figure(figsize=(7, 7))

    

    for i in range(count):

        label = labels[i]

        image = images[i].reshape([28,28])

        sub = plt.subplot(count_row_col, count_row_col, i + 1)

        sub.set_xticks(())

        sub.set_yticks(())

        sub.text(1, 1, label, ha='center', va='top', size=8, color="k")

        

        if not(predictions is None):

            pred = predictions[i]

            sub.text(25, 1, pred, ha='center', va='top', size=8, color="r")

        

        sub.imshow(image, cmap=plt.get_cmap('gray_r'))

    

    fig.tight_layout(pad = 0)

    plt.show()

    

def plot_performance(sample_range, learning_rates, 

                     train_accuracy = None, train_loss = None, 

                     test_accuracy = None, test_loss = None):

    

    plot_performance_chart(sample_range, test_accuracy, learning_rates, 'Test Accuracy')

    plot_performance_chart(sample_range, train_accuracy, learning_rates, 'Train Accuracy')

    plot_performance_chart(sample_range, test_loss, learning_rates, 'Test Loss', 'upper right')

    plot_performance_chart(sample_range, train_loss, learning_rates, 'Train Loss', 'upper right')

    

def plot_performance_chart(X, Ys, legend_labels, label, loc = 'lower right'):

    

    COLORS = ['r', 'g', 'b', 'm', 'c', 'y']

    

    if not(Ys is None):

        

        plt.figure(figsize=(10,4))

        

        for i in range(len(legend_labels)):

            plt.plot(X, Ys[i], c = COLORS[i], label = legend_labels[i])

            plt.legend(loc = loc, frameon = True, prop = {'size':7}, title = "Learning Rate")

            

            #plt.ylim(ymax = 3000, ymin = 2600)

            #plt.xlim(xmax = 100, xmin = 50)



        plt.ylabel(label)

        plt.xlabel('Steps')

        plt.show()
# visualizing the first 100th images and their labels

plot_batch_images(X_train[:100,:], np.argmax(Y_train[:100], 1))
# the batch size equals to BATCH_SIZE it will

# starts from 0 to until the last index of

# batch equals to TOTAL_COUNT

BATCH_COUNTER = 0



def next_batch(all_images, all_labels):

    

    # accessing BATCH_COUNTER inside function

    global BATCH_COUNTER

    

    # check if exceeds to the last index then reset batch_counter

    if BATCH_COUNTER == len(all_labels) / BATCH_SIZE:

        BATCH_COUNTER = 0



    # setting first and last index of data

    index_from = BATCH_COUNTER * BATCH_SIZE

    index_to   = (BATCH_COUNTER + 1) * BATCH_SIZE

    

    # loading a batch of training images and labels

    image_batch = all_images[index_from:index_to]

    label_batch = all_labels[index_from:index_to]

    

    # incrementing batch_counter for the next iteration

    BATCH_COUNTER += 1 

    

    return image_batch, label_batch;
def train(train_X, train_Y, test_X, test_Y, pred_X,

          learning_rate, iteration_count, accuracy_check = 0):

    

    # initialisation

    tf.set_random_seed(0)

    X = tf.placeholder(tf.float32, shape=[None, 784])

    W = tf.Variable(tf.zeros([784,10]))

    B = tf.Variable(tf.zeros([10]))

    init = tf.global_variables_initializer()

   

    # model

    Y = tf.nn.softmax(tf.matmul(X, W) + B)



    # placeholder for correct answers

    Y_ = tf.placeholder(tf.float32, [None, 10])

    

    # loss function

    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))



    # % of correct answers found in batch

    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))

    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_step = optimizer.minimize(cross_entropy)



    # start processing session

    sess = tf.Session()

    sess.run(init)



    for i in range(iteration_count):



        # loading a batch of training images and labels

        batch_X, batch_Y = next_batch(train_X, train_Y)



        # training the model with image and label batches

        sess.run(train_step, feed_dict = {X: batch_X, Y_: batch_Y})

    

        # checing the accuracy of our precess 

        # in every ACCURACY_CHECK iterations



        if (accuracy_check != 0) and (i % accuracy_check == 0):

            print('{}, '.format(i), end = '')

        

            # evaluating model for the trained data in every ACCURACY_CHECK count

            pred, a , c = sess.run([Y, accuracy, cross_entropy], feed_dict = {X: batch_X, Y_: batch_Y})

            print('Train(ACC: {:0.2f}, CE:{:0.3f})'.format(a, c), end = '')

        

            # evaluate test result in every ACCURACY_CHECK count

            pred, a , c = sess.run([Y, accuracy, cross_entropy], feed_dict = {X: test_X, Y_: test_Y})

            print(' - Test(AC: {:0.2f}, CE:{:0.3f})'.format(a, c))

    

    if accuracy_check != 0:

        acc = accuracy.eval(session=sess, feed_dict = {X: test_X, Y_: test_Y})

        print('Test set result = Accuracy: {:0.4f}'.format(acc))

    

    acc = accuracy.eval(session=sess, feed_dict = {X: train_X, Y_: train_Y})

    print('Train set result = Accuracy: {:0.4f}'.format(acc))

    

    if not(pred_X is None):

        pred = sess.run([Y], feed_dict = {X: pred_X})

    

    # closing the processing session

    sess.close()

    

    if not(pred_X is None):

        return pred[0]

    else:

        return 
train(X_train, Y_train, X_test, Y_test, None, LEARNING_RATE, ITERATION_COUNT, ACCURACY_CHECK)
Y_pred = train(X_train, Y_train, X_test, Y_test, X_pred, LEARNING_RATE, ITERATION_COUNT, 0)
predition = np.argmax(Y_pred[:len(Y_pred)], 1)

# visualizing the first 100th images and their predicted labels

plot_batch_images(X_pred[:100,:], predition[:100])
#prediction = np.argmax(Y_pred[:len(Y_pred)], 1)

# predictive model

#test_submission = model.predict_classes(test_data, verbose=2)



# save submission to csv

pd.DataFrame({"ImageId": list(range(1,len(predition)+1)), 

              "Label": predition}).to_csv('MNIST-submission_1-3-2017.csv', index=False,header=True)