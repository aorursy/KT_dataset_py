# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
im_height = 144

im_width = 192



im_height = 300

im_width = 360
train_labels = np.array(pd.read_csv("../input/train_labels.csv"))



train_folder = "../input/train"

train_list = check_output(["ls", train_folder]).decode("utf8").split()

train_data = np.zeros(shape=(len(train_list), im_height,im_width,3), dtype=np.float32) #np.uint8



for jpg in train_list:

    im_num = int(jpg.split('.')[0]) - 1

    im = Image.open(train_folder+"/"+jpg).resize((im_width,im_height))#,Image.ANTIALIAS)

    train_data[im_num] = np.array(im)/255
print(train_data[1])
#test_folder = "../input/test"

#test_list = check_output(["ls", test_folder]).decode("utf8").split()

#test_data = np.zeros(shape=(len(test_list), im_height,im_width,3), dtype=np.uint8)



#for jpg in test_list:

#    im_num = int(jpg.split('.')[0]) - 1

#    im = Image.open(test_folder+"/"+jpg).resize((im_width,im_height),Image.ANTIALIAS)

#    test_data[im_num] = np.array(im)
np.random.seed(42)

shuffle = np.random.permutation(np.arange(train_data.shape[0]))

train_data, train_labels = train_data[shuffle], train_labels[shuffle]



train_data_sub, train_labels_sub = train_data[:1800], train_labels[:1800]

dev_data, dev_labels = train_data[1800:], train_labels[1800:]
#print(train_labels_sub[:,1])
#from keras.models import Sequential, Model, load_model

#from keras import applications

#from keras import optimizers

#from keras.layers import Dropout, Flatten, Dense



#img_rows, img_cols, img_channel = 192, 144, 3



#base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
from keras import applications, losses, optimizers

from keras.models import Sequential, Model, load_model

from keras.layers import Dropout, Flatten, Dense, Activation

from keras.utils.np_utils import to_categorical



model = Sequential()

model.add(Flatten(input_shape=(im_height,im_width,3)))



model.add(Dense(units=2))

model.add(Activation('softmax'))

model.compile(loss=losses.categorical_crossentropy,

              optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),

              metrics=['accuracy']

             )



model.fit(train_data_sub, to_categorical(train_labels_sub[:,1]), epochs=2, batch_size=600, verbose=1)



#model.train_on_batch(train_data_sub, train_labels_sub)



#loss_and_metrics = model.evaluate(dev_data, to_categorical(dev_labels[:,1]), batch_size=360)



#classes = model.predict(test_data, batch_size=360)
import keras.layers as kl
print(train_data_sub.shape)
model = Sequential()

#model.add(kl.Flatten(input_shape=(im_height,im_width,3)))

model.add(kl.Conv2D(filters=64, kernel_size=3,strides=1, padding='same',activation='relu',input_shape=(im_height,im_width,3)))

model.add(kl.Conv2D(filters=32, kernel_size=3,strides=1, padding='same',activation='relu'))#,input_shape=(im_height,im_width,3))

model.add(kl.MaxPooling2D(pool_size=(2, 2), padding='valid'))          

#model.add(kl.Reshape(32 * 150 * 180))

model.add(kl.Dropout(0.25))

model.add(kl.Dense(32))

model.add(kl.Dropout(0.25))

model.add(kl.Dense(2))

model.add(kl.Activation('softmax'))



model.compile(loss=losses.categorical_crossentropy,

              optimizer=optimizers.Adam(),

              metrics=['accuracy']

             )



model.fit(x=train_data_sub, y=to_categorical(train_labels_sub[:,1]), epochs=25, batch_size=50, verbose=1)
import tensorflow as tf



from tensorflow.python.framework import random_seed



def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)
height = 866

width = 1154

channels = 3

n_inputs = height * width



conv1_fmaps = 64

conv1_ksize = 3

conv1_stride = 1

conv1_pad = "SAME"



conv2_fmaps = 32

conv2_ksize = 3

conv2_stride = 1

conv2_pad = "SAME"

conv2_dropout_rate = 0.25





pool3_fmaps = conv2_fmaps



n_fc1 = 32

fc1_dropout_rate = 0.25



n_outputs = 2



reset_graph()



with tf.name_scope("inputs"):

    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")

    X_resized = tf.image.resize_images(X, [300, 360])



    y = tf.placeholder(tf.int32, shape=[None], name="y")

    training = tf.placeholder_with_default(False, shape=[], name='training')





conv1 = tf.layers.conv2d(X_resized, filters=conv1_fmaps, kernel_size=conv1_ksize,

                         strides=conv1_stride, padding=conv1_pad,

                         activation=tf.nn.relu, name="conv1")

conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,

                         strides=conv2_stride, padding=conv2_pad,

                         activation=tf.nn.relu, name="conv2")



with tf.name_scope("pool3"):

    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 150 * 180])

    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):

    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)



with tf.name_scope("output"):

    logits = tf.layers.dense(fc1, n_outputs, name="output")

    Y_proba = tf.nn.softmax(logits, name="Y_proba")



with tf.name_scope("train"):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer()

    training_op = optimizer.minimize(loss)



with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



with tf.name_scope("init_and_save"):

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
y_test = train_labels[2200:]

test_data = DataSet(np.arange(2200,len(train_labels)), y_test)

train_data =  DataSet(np.arange(0,2200), train_labels[:2200])

X_val, y_val = test_data.next_batch(70)









n_epochs = 25

batch_size = 50



best_loss_val = np.infty

check_interval = 5

checks_since_last_progress = 0

max_checks_without_progress = 100

best_model_params = None 



with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        for iteration in range(train_data.num_examples // batch_size):

            X_batch, y_batch = train_data.next_batch(batch_size)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

            if iteration % check_interval == 0:

                loss_val = loss.eval(feed_dict={X: X_val,

                                                y: y_val})

                if loss_val < best_loss_val:

                    best_loss_val = loss_val

                    checks_since_last_progress = 0

                    best_model_params = get_model_params()

                else:

                    checks_since_last_progress += 1

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: X_val, y: y_val})

        loss_val = loss.eval(feed_dict={X: X_val,

                                                y: y_val})

        print("Epoch {}, train accuracy: {:.4f}%, test accuracy: {:.4f}%, loss: {:.6f}, best loss: {:.6f}".format(

                  epoch, acc_train * 100, acc_test * 100, loss_val, best_loss_val))

        if checks_since_last_progress > max_checks_without_progress:

            print("Early stopping!")

            break



    if best_model_params:

        restore_model_params(best_model_params)



    save_path = saver.save(sess, "./my_isd_model2")
from keras.layers import convolutional



height = 866

width = 1154

channels = 3

n_inputs = height * width



conv1_fmaps = 64

conv1_ksize = 3

conv1_stride = 1

conv1_pad = "SAME"



conv2_fmaps = 32

conv2_ksize = 3

conv2_stride = 1

conv2_pad = "SAME"

conv2_dropout_rate = 0.25



pool3_fmaps = conv2_fmaps



n_fc1 = 32

fc1_dropout_rate = 0.25



n_outputs = 2



conv1 = convolutional.Conv2d(train_data, filters=conv1_fmaps, kernel_size=conv1_ksize,

                         strides=conv1_stride, padding=conv1_pad,

                         activation=Activation("relu"), name="conv1")

conv2 = convolutional.Conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,

                         strides=conv2_stride, padding=conv2_pad,

                         activation=Activation("relu"), name="conv2")



convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 

                     data_format=None, dilation_rate=(1, 1), activation=None, 

                     use_bias=True, kernel_initializer='glorot_uniform', 

                     bias_initializer='zeros', kernel_regularizer=None, 

                     bias_regularizer=None, activity_regularizer=None, 

                     kernel_constraint=None, bias_constraint=None)