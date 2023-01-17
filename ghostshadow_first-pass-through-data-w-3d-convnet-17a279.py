import os

import pandas as pd

import dicom

import cv2

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



IMG_PX_SIZE = 64

FC_SIZE = 16 * 16 * 16 * 8



N_CLASSES = 2

BATCH_SIZE = 10

HM_EPOCHS = 10



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)
def process_data(patient,labels_df):    

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))



    # Resize the x-y axis

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    slices = np.array(slices)



    # Rotate the cuboid 

    slices = np.transpose(slices,[2,1,0])



    # Resize the new x-y axis

    new_slices = []

    for i in range(len(slices)):

        new_slices.append(cv2.resize(slices[i],(IMG_PX_SIZE,IMG_PX_SIZE)))

    new_slices = np.array(new_slices)    



    # Rotate back (Optional: Conv3D doesn't care)

    new_slices = np.transpose(new_slices,[2,1,0])



    #print(slices.shape,new_slices.shape)



    # Normalize

    new_slices = np.array(new_slices,dtype='float32')

    new_slices -= new_slices.min()

    new_slices /= new_slices.max()

    #print(new_slices.mean())

    

    return new_slices,label
much_data = []

print('Total Patients: ',len(patients))



file_name = 'muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,IMG_PX_SIZE)



if os.path.isfile(file_name):

    print('File found, loaded')

    much_data = np.load(file_name)

else:

    print('File not found, creating')

    for num,patient in enumerate(patients):

        if num % 100 == 0:

            print(num)

        try:

            img_data,label = process_data(patient,labels_df)

            much_data.append([img_data,label])

        except KeyError as e:

            print('This is unlabeled data!')

    np.save(file_name, much_data)

    print('Saved')
# Added shapes

x = tf.placeholder('float',[None,IMG_PX_SIZE,IMG_PX_SIZE,IMG_PX_SIZE])

y = tf.placeholder('float',[None,N_CLASSES])



keep_rate = 0.8



def conv3d(x, W):

    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')



def maxpool3d(x):

    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def convolutional_neural_network(x):

    # Changed the variables

    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,4])),

               'W_conv2':tf.Variable(tf.random_normal([3,3,3,4,8])),

               'W_fc':tf.Variable(tf.random_normal([FC_SIZE,1024])),

               'out':tf.Variable(tf.random_normal([1024, N_CLASSES]))}



    biases = {'b_conv1':tf.Variable(tf.random_normal([4])),

               'b_conv2':tf.Variable(tf.random_normal([8])),

               'b_fc':tf.Variable(tf.random_normal([1024])),

               'out':tf.Variable(tf.random_normal([N_CLASSES]))}



    #                            image X      image Y        image Z

    x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, IMG_PX_SIZE, 1])



    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])

    conv1 = maxpool3d(conv1)



    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])

    conv2 = maxpool3d(conv2)

    

    print(conv2.get_shape())



    fc = tf.reshape(conv2,[-1, FC_SIZE])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    fc = tf.nn.dropout(fc, keep_rate)



    output = (tf.matmul(fc, weights['out'])+biases['out'])



    return output
validation_size = int(.2 * len(much_data))



trainX = []

trainY = []

validateX = []

validateY = []



for data in much_data[:-validation_size]:

    trainX.append(data[0])

    trainY.append(np.eye(N_CLASSES)[data[1]])



trainX = np.array(trainX)

trainY = np.array(trainY)



for data in much_data[-validation_size:]:

    validateX.append(data[0])

    validateY.append(np.eye(N_CLASSES)[data[1]])



validateX = np.array(validateX)

validateY = np.array(validateY)
def train_neural_network(x):    

    prediction = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)    

    

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        

        successful_runs = 0

        total_runs = 0

        

        for epoch in range(HM_EPOCHS):

            epoch_loss = 0

            try:

                indices = [np.random.randint(len(trainX)) for _ in range(BATCH_SIZE)]

                indices = list(set(indices))

                

                batchX = np.take(trainX,indices,0)

                batchY = np.take(trainY,indices,0)

                _, c = sess.run([optimizer, cost], feed_dict={x: batchX, y: batchY})

                epoch_loss += c

            except Exception as e:                    

                print(str(e))

            

            print('Epoch', epoch+1, 'completed out of',HM_EPOCHS,

                  'Batch size: ',len(indices),'loss:',epoch_loss)



            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))



            print('Accuracy:',accuracy.eval({x:validateX, y:validateY}))

            

        print('Done. Finishing accuracy:')

        print('Accuracy:',accuracy.eval({x:validateX, y:validateY}))



train_neural_network(x)