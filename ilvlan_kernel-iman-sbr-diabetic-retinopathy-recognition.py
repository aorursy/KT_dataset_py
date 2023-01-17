import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 

import matplotlib.pyplot as plt

import os

# %load_ext tensorboard.notebook

# %tensorboard --logdir logs



image_dim = 128

def read_label_filepath(path):

    data = pd.read_csv(path)

    labels = data.iloc[:,[1]].values

    zeros = np.zeros((len(labels),5))

    for i in range(len(labels)): 

        zeros[i,labels[i]]=1 

    labels = zeros    

    filename = data.iloc[:,[0]].values

    trian_path ='../input/diabetic-retinopathy-resized/resized_train/resized_train/' + filename + '.jpeg'

    return labels , trian_path





def read_images_from_disk(path):

    file_contents = tf.read_file(path)

    img_tensor = tf.image.decode_jpeg(file_contents, channels=3)

    img_final = tf.image.resize_images(img_tensor, (image_dim, image_dim))

    img_final = tf.cast(img_final, tf.uint8)

    img_final = tf.image.rgb_to_grayscale(img_final)

    return img_final





train_labels,train_filepath = read_label_filepath('../input/diabetic-retinopathy-resized/trainLabels.csv')

train_filepath=train_filepath.flatten()









label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int16))

path_ds = tf.data.Dataset.from_tensor_slices(train_filepath)

image_ds = path_ds.map(read_images_from_disk, num_parallel_calls=10)

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))





labels_count = np.zeros(shape = 5)

labels_iterator = label_ds.make_one_shot_iterator()

next_element = labels_iterator.get_next()

with tf.Session() as sess:

    while True:

        try:

            lbl = sess.run(next_element)

            labels_count += lbl

#             print(labels_count)

        except tf.errors.OutOfRangeError:

            break



plt.bar(range(5),height = labels_count)

plt.ylabel("number of labels in each category")

plt.xlabel("labels")

plt.show()

# Any results you write to the current directory are saved as output.
def init_weight_dist(shape):

    weights = tf.truncated_normal(shape,stddev=0.1)

    return tf.Variable(weights)



def init_bias_vals(shape):

    biases = tf.constant(1.0,shape=shape)

    return tf.Variable(biases)

def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')



def pooling(x):

    return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')



def convolutional_layer(input_x,shape):

    w = init_weight_dist(shape)

    b = init_bias_vals([shape[3]])

    return tf.nn.relu(conv2d(input_x,w)+b)



def normal_full_layer(input_layer,size):

    input_size = int(input_layer.get_shape()[1])

    W = init_weight_dist([input_size,size])

    b = init_bias_vals([size])

    return tf.matmul(input_layer,W) + b
image_dim = 128

x = tf.placeholder(tf.float32,shape=[None,image_dim,image_dim,1])

y_true = tf.placeholder(tf.int32,shape=[None,5])

hold_prob = tf.placeholder(tf.float32)



x_image = tf.reshape(x,[-1,image_dim,image_dim,1])

print(x_image.shape)

conv1 = convolutional_layer(x_image,shape=[50,50,1,32])

conv1_pool = pooling(conv1)



conv2 = convolutional_layer(conv1_pool,shape=[25,25,32,8])

conv2_pool = pooling(conv2)

conv2_flat = tf.reshape(conv2_pool,[-1,51200])



full_layer_one = tf.nn.relu(normal_full_layer(conv2_flat,128))

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,5)



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true,logits = y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(cross_entropy)



init = tf.global_variables_initializer()



batch_data = image_label_ds.batch(100)

iterator = batch_data.make_initializable_iterator()

next_element = iterator.get_next()

# import matplotlib.image as mpimg

from PIL import Image

import matplotlib.pyplot as plt

true_wm=Image.open('../input/true-false-img/true.png').resize((128,128), Image.ANTIALIAS)

false_wm=Image.open('../input/true-false-img/false.png').resize((128,128), Image.ANTIALIAS)

def show_images(batch,matches):

    plt.rcParams["figure.figsize"]=20,20

    count = len(batch)

    sq = int(np.sqrt(count))

    fig, ax = plt.subplots(sq, sq)

    

    for i in range(sq):

        for j in range(sq):

            ax[i,j].imshow(np.asarray(batch[i*sq + j]).reshape((128,128)),cmap='gray')

            if matches[i*sq + j] == True:

               ax[i,j].imshow(true_wm, aspect='auto', zorder=1, alpha=0.7)

            else:

               ax[i,j].imshow(false_wm, aspect='auto', zorder=1, alpha=0.7)  

    plt.show()
num_epoch = 3

show_best = False

with tf.Session() as sess:

    sess.run(init)

    for i in range(num_epoch):

        sess.run(iterator.initializer)

        acc_sum = 0

        j = 0

        while True:

            try:

                batch_x , batch_y  = sess.run(next_element)

#                 print("epoch {}, batch {}: ".format(i+1,j+1) +  str(batch_x.shape))

                sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob: 0.5})

                matches = tf.equal(tf.arg_max(y_pred,1),tf.argmax(y_true,1))

                acc = tf.reduce_mean(tf.cast(matches,tf.float32))

                

                

                acc_res,matches = sess.run([acc,matches],feed_dict={x:batch_x,y_true : batch_y,hold_prob: 1.0})

                acc_sum += acc_res

                

#                 print("acc: " + str(acc_res))

                if acc_res > 0.9 and show_best == False:

                    show_best = True

                    print("just for good visualization:")

                    print("the best batch detection with accuracy {}".format(str(acc_res)))

                    show_images(batch_x[0:100],matches[0:100])

                j += 1

            except tf.errors.OutOfRangeError:   

                break

            except tf.errors.InvalidArgumentError:

                break

        print("epoch accuracy :" + str(acc_sum/j))

    writer = tf.summary.FileWriter('../output/graphs', sess.graph)