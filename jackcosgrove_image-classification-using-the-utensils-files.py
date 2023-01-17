import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf



# settings

LEARNING_RATE = 1e-4



TRAINING_ITERATIONS = 1000        

    

DROPOUT = 0.5

BATCH_SIZE = 40



# set to 0 to train on all available data

VALIDATION_SIZE = 80



# size of each image dimension

IMAGE_DIMENSION_SIZE = 64
# read training data from CSV file 

data = pd.read_csv('../input/utensils'+str(IMAGE_DIMENSION_SIZE)+'x'+str(IMAGE_DIMENSION_SIZE)+'_train.csv')



print('data({0[0]},{0[1]})'.format(data.shape))
# read labels from CSV file

labels_csv = pd.read_csv('../input/utensils_labels.csv')

labels = {}

for index, row in labels_csv.iterrows():

    labels[row['Label']] = row['Name']



for k, v in labels.items():

    print(k, v)
images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



print('images({0[0]},{0[1]})'.format(images.shape))
image_size = images.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
# display images

def display(data, num_per_group):

    

    rows = data.sample(frac=1).groupby('Label', sort=False).head(num_per_group)

    

    figure_width = figure_height = np.ceil(np.sqrt(rows.shape[0])).astype(np.uint8)

    

    fig = plt.figure(figsize=(8, 8))

    

    i = 1

    for index, row in rows.iterrows():

        one_image = row[1:].values.reshape(image_width,image_height)

        label = labels[row[0]]

        sub = fig.add_subplot(figure_height, figure_width, i)

        sub.axis('off')

        sub.set_title(label)

        sub.imshow(one_image, cmap=cm.binary)

        i += 1



# output image     

display(data, 4)
labels_flat = data.iloc[:,0].values.ravel()



print('labels_flat({0})'.format(len(labels_flat)))
labels_count = np.unique(labels_flat).shape[0]



print('labels_count => {0}'.format(labels_count))
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('labels({0[0]},{0[1]})'.format(labels.shape))
# split data into training & validation

validation_images = images[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = images[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]





print('train_images({0[0]},{0[1]})'.format(train_images.shape))

print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))
# weight initialization

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
# convolution

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# input & output of NN



# images

x = tf.placeholder('float', shape=[None, image_size])

# labels

y_ = tf.placeholder('float', shape=[None, labels_count])
# first convolutional layer

W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])



image = tf.reshape(x, [-1,image_width , image_height,1])



h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)



h_pool1 = max_pool_2x2(h_conv1)





# Prepare for visualization

# display 32 fetures in 4 by 8 grid

layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))  



# reorder so the channels are in the first dimension, x and y follow.

layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))



layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8)) 
# second convolutional layer

W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



h_pool2 = max_pool_2x2(h_conv2)





# Prepare for visualization

# display 64 features in 4 by 16 grid

layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))  



# reorder so the channels are in the first dimension, x and y follow.

layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))



layer2 = tf.reshape(layer2, (-1, 14*4, 14*16)) 
# densely connected layer

W_fc1 = weight_variable([IMAGE_DIMENSION_SIZE*IMAGE_DIMENSION_SIZE*4, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_DIMENSION_SIZE*IMAGE_DIMENSION_SIZE*4])



h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout

keep_prob = tf.placeholder('float')

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer for deep net

W_fc2 = weight_variable([1024, labels_count])

b_fc2 = bias_variable([labels_count])



y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# cost function

cross_entropy = -tf.reduce_sum(y_*tf.log(y))





# optimisation function

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)



# evaluation

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# prediction function

#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1

predict = tf.argmax(y,1)
epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return train_images[start:end], train_labels[start:end]
# start TensorFlow session

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()



sess.run(init)
# visualisation variables

train_accuracies = []

validation_accuracies = []

x_range = []



display_step=1



for i in range(TRAINING_ITERATIONS):



    #get new batch

    batch_xs, batch_ys = next_batch(BATCH_SIZE)        



    # check progress on every 1st,2nd,...,10th,20th,...,100th... step

    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:

        

        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 

                                                  y_: batch_ys, 

                                                  keep_prob: 1.0})       

        if(VALIDATION_SIZE):

            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 

                                                            y_: validation_labels[0:BATCH_SIZE], 

                                                            keep_prob: 1.0})                                  

            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))

            

            validation_accuracies.append(validation_accuracy)

            

        else:

             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))

        train_accuracies.append(train_accuracy)

        x_range.append(i)

        

        # increase display_step

        if i%(display_step*10) == 0 and i:

            display_step *= 10

    # train on batch

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
# check final accuracy on validation set  

if(VALIDATION_SIZE):

    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 

                                                   y_: validation_labels, 

                                                   keep_prob: 1.0})

    print('validation_accuracy => %.4f'%validation_accuracy)

    plt.plot(x_range, train_accuracies,'-b', label='Training')

    plt.plot(x_range, validation_accuracies,'-g', label='Validation')

    plt.legend(loc='lower right', frameon=False)

    plt.ylim(ymax = 1.1, ymin = 0.3)

    plt.ylabel('accuracy')

    plt.xlabel('step')

    plt.show()
# read test data from CSV file 

test_images = pd.read_csv('../input/utensils'+str(IMAGE_DIMENSION_SIZE)+'x'+str(IMAGE_DIMENSION_SIZE)+'_test.csv')

test_images = test_images.iloc[:,1:].values

test_images = test_images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

test_images = np.multiply(test_images, 1.0 / 255.0)



print('test_images({0[0]},{0[1]})'.format(test_images.shape))





# predict test set

predicted_lables = np.zeros(test_images.shape[0])

for i in range(0,test_images.shape[0]//BATCH_SIZE):

    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 

                                                                                keep_prob: 1.0})





print('predicted_lables({0})'.format(len(predicted_lables)))



# save results

np.savetxt('submission_softmax.csv', 

           np.c_[range(1,len(test_images)+1),predicted_lables], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')
sess.close()