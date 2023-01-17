import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import time

import os

from PIL import Image



IMAGE_HEIGHT = 32

IMAGE_WIDTH = 32

IMAGE_CHANNELS = 3



data_path = '../input/fruits-360_dataset/fruits-360/'



def createTFRecordsFile(src_dir,tfrecords_name):

    dir = src_dir

    writer = tf.python_io.TFRecordWriter(tfrecords_name)



    samples_size = 0

    index = -1

    classes_dict = {}



    for folder_name in os.listdir(dir):

        class_path = dir + '/' + folder_name + '/'

        # class_path = dir+'\\'+folder_name+'\\'

        index +=1

        classes_dict[index] = folder_name

        # print(index, folder_name)

        for image_name in os.listdir(class_path):

            image_path = class_path+image_name

            # print(image_path)

            img = Image.open(image_path)

            img = img.resize((IMAGE_HEIGHT,IMAGE_WIDTH))

            img_raw = img.tobytes()

            example = tf.train.Example(

                features = tf.train.Features(

                    feature = {

                        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),

                        # 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(index)])),

                        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))

                    }

                )

            )

            writer.write(example.SerializeToString())

            samples_size +=1

    writer.close()

    print("totally %i samples" %samples_size)

    print(classes_dict)

    return  samples_size,classes_dict





def decodeTFRecordsFile(tfrecords_name):

    file_queue = tf.train.string_input_producer([tfrecords_name])

    reader = tf.TFRecordReader()

    _,serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(

        serialized_example,

        features = {

            'label':tf.FixedLenFeature([],tf.int64),

            'image_raw':tf.FixedLenFeature([],tf.string)

        }

    )

    img = tf.decode_raw(features['image_raw'],tf.uint8)

    img = tf.reshape(img,[IMAGE_HEIGHT,IMAGE_WIDTH,3])

    img = tf.cast(img,tf.float32)*(1./255)-0.5

    label = tf.cast(features['label'], tf.int32)



    return  img,label



def inputs(tfrecords_name,batch_size, shuffle = True):

    image,label = decodeTFRecordsFile(tfrecords_name)

    if(shuffle):

        images,labels = tf.train.shuffle_batch([image,label],

                                               batch_size=batch_size,

                                               capacity=train_samples_size+batch_size,

                                               min_after_dequeue=train_samples_size)

    else:

        # input_queue = tf.train.slice_input_producer([image, label], shuffle=False)

        images, labels = tf.train.batch([image,label],

                                        batch_size=batch_size,

                                        capacity=batch_size*2)

    return images,labels



def createFruitTrainTFRecordsFile():

    src_dir = data_path+'Training'

    # src_dir = 'fruits_360_dataset_2018_01_02\Training'

    print('createFruitTrainTFRecordsFile',src_dir)

    tfrecords_name = 'fruits_train.tfrecords'

    samples_size, fruits_dict = createTFRecordsFile(src_dir=src_dir,tfrecords_name=tfrecords_name)

    print('createFruitTrainTFRecordsFile done')

    return samples_size, fruits_dict



def createFruitTestTFRecordsFile():

    src_dir = data_path+'Test'

    # src_dir = 'fruits_360_dataset_2018_01_02\Validation'

    print('createFruitTestTFRecordsFile',src_dir)

    tfrecords_name = 'fruits_test.tfrecords'

    samples_size, fruits_dict = createTFRecordsFile(src_dir=src_dir,tfrecords_name=tfrecords_name)

    print('createFruitTestTFRecordsFile done')

    return samples_size, fruits_dict



train_samples_size, fruits_dict = createFruitTrainTFRecordsFile()#19426

test_samples_size, fruits_dict = createFruitTestTFRecordsFile()#6523



# for test only

# test_samples_size = 6523

# train_samples_size = 19426

# fruits_dict = {0: 'Apple Red 1', 1: 'Apple Red 2', 2: 'Apple Red 3', 3: 'Apple Red Delicious', 4: 'Apple Red Yellow', 5: 'Apricot',

#                6: 'Avocado', 7: 'Avocado ripe', 8: 'Braeburn', 9: 'Cactus fruit', 10: 'Carambula', 11: 'Cherry', 12: 'Clementine',

#                13: 'Cocos', 14: 'Golden 1', 15: 'Golden 2', 16: 'Golden 3', 17: 'Granadilla', 18: 'Granny Smith', 19: 'Grape Pink',

#                20: 'Grape White', 21: 'Grapefruit', 22: 'Kaki', 23: 'Kiwi', 24: 'Kumquats', 25: 'Lemon', 26: 'Limes', 27: 'Litchi',

#                28: 'Mango', 29: 'Nectarine', 30: 'Orange', 31: 'Papaya', 32: 'Passion Fruit', 33: 'Peach', 34: 'Peach Flat',

#                35: 'Pear', 36: 'Pear Monster', 37: 'Plum', 38: 'Pomegranate', 39: 'Quince', 40: 'Strawberry'}







# parameters

lr_start = 0.001

lr_end = 0.0001

learning_rate = lr_start



num_steps = 8000

batch_size = 64

update_step = 5

display_step = 100

train_acc_target = 1

train_acc_target_cnt = train_samples_size/batch_size

# if train_acc_target_cnt>20:

#     train_acc_target_cnt = 20



# network parameters

num_input = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS

num_classes = len(fruits_dict)

dropout = 0.5



# saver train parameters

useCkpt = False

checkpoint_step = 5

checkpoint_dir = os.getcwd()+'\\checkpoint\\'



# tf graph input

X = tf.placeholder(tf.float32,[None,num_input])

Y = tf.placeholder(tf.float32,[None,num_classes])

keep_prob = tf.placeholder(tf.float32)







# create some wrappers for simplicity

def conv2d(x,W,b,strides=1):

    # conv2d wrapper, with bias and relu activation

    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')

    # x = tf.nn.conv3d(x,W,strides=[1,strides,strides,strides,1],padding='SAME')

    x = tf.nn.bias_add(x,b)

    return tf.nn.relu(x)



def maxpool2d(x,k=2):

    # max2d wrapper

    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')



def conv_net(X,weights,biases,dropout):

    X = tf.reshape(X, shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])



    # convolustion layer

    conv1 = conv2d(X,weights['wc1'],biases['bc1'])

    # max pooling (down-sampling)

    conv1 = maxpool2d(conv1, k=2)



    # convolustion layer

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # max pooling (down-sampling)

    conv2 = maxpool2d(conv2, k=2)

    

    # apply dropout

    # conv2 = tf.nn.dropout(conv2, 0.98)



    # convolustion layer

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    # max pooling (down-sampling)

    conv3 = maxpool2d(conv3, k=2)

    

    # apply dropout

    # conv3 = tf.nn.dropout(conv3, 0.95)



    # convolustion layer

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])

    # max pooling (down-sampling)

    conv4 = maxpool2d(conv4, k=2)

    

    # apply dropout

    # conv4 = tf.nn.dropout(conv4, 0.9)



    # convolustion layer

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])

    # max pooling (down-sampling)

    conv5 = maxpool2d(conv5, k=2)

    

    # apply dropout

    conv5 = tf.nn.dropout(conv5, 0.9)



    # print(conv4.shape)

    # fully connected layer

    fc1 = tf.reshape(conv5, shape=[-1,weights['wd1'].get_shape().as_list()[0]])

    # print('conv4 shape:', conv4.shape, ', fc1 shape:', fc1.shape)

    fc1 = tf.add(tf.matmul(fc1,weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)



    # apply dropout

    fc1 = tf.nn.dropout(fc1, dropout)



    # output, class prediction

    out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])

    return  out



# store layers weighta and bias

weights = {

    # 5x5 conv, 3 inputs, 16 outpus

    'wc1': tf.get_variable('wc1',[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    # 5x5 conv, 16 input, 32 outpus

    'wc2': tf.get_variable('wc2',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    # 5x5 conv, 32 inputs, 64 outputs

    'wc3': tf.get_variable('wc3',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    # 5x5 conv, 64 inputs, 128 outputs

    'wc4': tf.get_variable('wc4',[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    # 5x5 conv, 128 inputs, 256 outputs

    'wc5': tf.get_variable('wc5', [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),



    # fully connected, 7*7*128 inputs, 2048 outputs

    'wd1': tf.get_variable('wd1',[1*1*512,2048],initializer=tf.contrib.layers.xavier_initializer()),

    # 32 inputs, 26 outputs (class prediction)

    'out': tf.get_variable('fc1',[2048,num_classes],initializer=tf.contrib.layers.xavier_initializer()),

}

biases = {

    'bc1': tf.Variable(tf.zeros([32])),

    'bc2': tf.Variable(tf.zeros([64])),

    'bc3': tf.Variable(tf.zeros([128])),

    'bc4': tf.Variable(tf.zeros([256])),

    'bc5': tf.Variable(tf.zeros([512])),

    'bd1': tf.Variable(tf.zeros([2048])),

    'out': tf.Variable(tf.zeros([num_classes]))

}





# cconstruct model

logits = conv_net(X,weights,biases,keep_prob)

prediction = tf.nn.softmax(logits)





# define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,

                                                                 labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss=loss_op)



# evaluate model

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))



# initialization

init = tf.global_variables_initializer()





def trainModel():

    acc_meet_target_cnt = 0

    tic = time.time()

    for step in range(1, num_steps + 1):

        with tf.Graph().as_default():

            if train_acc_target_cnt <= acc_meet_target_cnt:

                break

            

            # batch_x, batch_y = train_next_batch(batch_size)

            # test batch

            batch_x, y = sess.run([images, labels])

            batch_y = np.zeros(shape=[batch_size, num_classes])

            for i in range(batch_size):

                batch_y[i, y[i]] = 1

            batch_x = np.reshape(batch_x, [batch_size, num_input])



            # run optimization op (backprop)

            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

           



            if step % update_step == 0 or step == 1:

                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})

                learning_rate = updateLearningRate(acc, lr_start=lr_start)

                if train_acc_target <= acc:

                    acc_meet_target_cnt += 1

                else:

                    acc_meet_target_cnt = 0

                toc = time.time()

                

                tic = toc

            if step % display_step == 0 or step == 1:

                print("{:.4f}".format(toc - tic)+ "s,", "step " + str(step) + ", minibatch loss = " + \

                      "{:.4f}".format(loss) + ", training accuracy = " + \

                      "{:.4f}".format(acc) , ", acc_meet_target_cnt = " + "{:.4f}".format(acc_meet_target_cnt))



            if useCkpt:

                if step % checkpoint_step == 0 or train_acc_target_cnt <= acc_meet_target_cnt:

                    # saver.save(sess,checkpoint_dir+'model.ckpt',global_step=step)

                    saver.save(sess, checkpoint_dir + 'model.ckpt')



def testModel(images, labels):

    # calulate the test data sets accuracy

    samples_untest = test_samples_size

    acc_sum = 0

    test_sample_sum = 0

    while samples_untest > 0:

        with tf.Graph().as_default():

            test_batch_size = batch_size

            # if (test_batch_size > samples_untest):

                # test_batch_size = samples_untest

            # test_images, test_labels = test_next_batch(batch_size=test_batch_size, shuffle=False)



            # test batch

            test_images, y = sess.run([images, labels])

            test_labels = np.zeros(shape=[test_batch_size, num_classes])

            for i in range(test_batch_size):

                test_labels[i, y[i]] = 1



            test_images = np.reshape(test_images, [test_batch_size, num_input])

            acc = sess.run(accuracy, feed_dict={X: test_images, Y: test_labels, keep_prob: 1})

            acc_sum += acc * test_batch_size

            # print("samples_untest = ", samples_untest, ", acc_current = ", acc)

            samples_untest -= test_batch_size

            test_sample_sum += test_batch_size

    print("Testing accuracy = ", \

          # sess.run(accuracy,feed_dict={X:mnist.test.images/255, Y:mnist.test.labels}))

          acc_sum / test_sample_sum)



def updateLearningRate(acc,lr_start):

    learning_rate_new = lr_start - acc*lr_start*0.9

    return learning_rate_new



saver = tf.train.Saver()



# start training

with tf.Session() as sess:

    # run the initailizer

    sess.run(init)



    # train batch

    tfrecords_name = 'fruits_train.tfrecords'

    images, labels = inputs(tfrecords_name, batch_size, shuffle = True)

    # create coord

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    if useCkpt:

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)

        else:

            pass



    # train the model

    trainModel()

    print("Optimization finish!")



    # test batch

    tfrecords_name = 'fruits_test.tfrecords'

    images, labels = inputs(tfrecords_name, batch_size, shuffle=False)

    # create coord

    coord2 = tf.train.Coordinator()

    threads2 = tf.train.start_queue_runners(sess=sess, coord=coord2)



    if useCkpt:

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:

            saver.restore(sess, ckpt.model_checkpoint_path)

        else:

            pass



    # test the model

    testModel(images, labels)



    # close coord

    coord.request_stop()

    coord.join(threads)

    coord2.request_stop()

    coord2.join(threads2)

    sess.close()