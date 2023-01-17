import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio

import matplotlib.pyplot as plt

import random

import time

from datetime import timedelta

import tensorflow as tf
image_paths = []

imseg_paths = []

for x in ['dataA', 'dataB', 'dataC', 'dataD', 'dataE']:

    image_path_dir = './lyft-udacity-challenge/' + x + '/' + x + '/' + 'CameraRGB'

    imseg_path_dir = './lyft-udacity-challenge/' + x + '/' + x + '/' + 'CameraSeg'



    for dirname, _, filenames in os.walk(image_path_dir):        

        for filename in filenames:

            image_path = image_path_dir + '/' + filename

            image_paths.append(image_path)

            imseg_path = imseg_path_dir + '/' + filename

            imseg_paths.append(imseg_path) 

            

# Number of images

num_images = len(image_paths)

print("Total number of images = ", num_images)
class Args:

    L2_REG = 1e-5

    STDEV = 1e-2

    KEEP_PROB = 0.8

    LEARNING_RATE = 1e-4

    EPOCHS = 1

    BATCH_SIZE = 32

    IMAGE_SHAPE = (600, 800)

    NUM_CLASSES = 2

    SEGMENT = 7 # Segment lable for road



args = Args()
# Read image

def read_image(image_path, imseg_path):

    # Output is the image and corrosponding segment of the road

    segment = args.SEGMENT

    

    height, width = args.IMAGE_SHAPE

    

    image = np.zeros((height, width, 3), dtype=np.int16)

    imseg = np.zeros((height, width, 1), dtype=np.int8)



    image = imageio.imread(image_path) # RGB image

    imseg = imageio.imread(imseg_path) # Segmented image 



    imseg_road = np.zeros((height, width, 1))

    imseg_road[np.where(imseg==segment)[0], np.where(imseg==segment)[1]] = 1.0

    

    return image, imseg_road
# Visualise the images

fig, axes = plt.subplots(3, 3, figsize=(30,20))

for num in range(3):

    N = random.randrange(len(image_paths))



    image, imseg_road = read_image(image_paths[N], imseg_paths[N])

    imseg_road = imseg_road.reshape((args.IMAGE_SHAPE))

    

    imseg = imageio.imread(imseg_paths[N])

    imseg = np.array([max(imseg[i, j]) for i in range(imseg.shape[0]) for j in range(imseg.shape[1])]).reshape(image.shape[0], image.shape[1])





    axes[num, 0].imshow(image)

    axes[num, 0].set_title('RGB Image')

    axes[num, 1].imshow(imseg)

    axes[num, 1].set_title('Segmented image')

    axes[num, 2].imshow(imseg_road)

    axes[num, 2].set_title('Segmented road image')

# Split the data into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_paths, imseg_paths, test_size=0.1, 

                                                    shuffle =True,  random_state=42)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, 

                                                    shuffle =True,  random_state=42)



print("Training images = ", len(X_train) ,"|", "Validation images = ", len(X_val), "|", "Test images = ", len(X_test))
# load vgg function

def load_vgg(sess, vgg_path):

    # Load the pre-trained model and weights

    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'

    vgg_keep_prob_tensor_name = 'keep_prob:0'

    vgg_layer3_out_tensor_name = 'layer3_out:0'

    vgg_layer4_out_tensor_name = 'layer4_out:0'

    vgg_layer7_out_tensor_name = 'layer7_out:0'

    

    tf.compat.v1.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    print(graph.get_operations())

    

    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)

    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)

    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)

    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    

    return img_input, keep, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    """

    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.

    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output

    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output

    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output

    :param num_classes: Number of classes to classify

    :return: The Tensor for the last layer of output

    """

    # Define the regularizer for the kernel

    kernel_regularizer = tf.contrib.layers.l2_regularizer(args.L2_REG)

    

    layer7_conv_1x1 = tf.layers.conv2d(inputs = vgg_layer7_out, filters = num_classes,

                                      kernel_size = (1,1), strides = (1,1),

                                      padding = "same", kernel_regularizer = kernel_regularizer)

    deconv_layer7 = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, (2,2), padding="same", kernel_regularizer = kernel_regularizer)

    

    layer4_conv_1x1 = tf.layers.conv2d(inputs = vgg_layer4_out, filters = num_classes,

                               kernel_size = (1,1), strides = (1,1),

                               padding = "same", kernel_regularizer = kernel_regularizer)

    skip_connection_1 = tf.add(deconv_layer7, layer4_conv_1x1)

    

    deconv_layer_4_7 = tf.layers.conv2d_transpose(skip_connection_1, num_classes, 4, (2,2), padding="same", kernel_regularizer = kernel_regularizer)

    

    layer3_conv_1x1 = tf.layers.conv2d(inputs = vgg_layer3_out, filters = num_classes,

                               kernel_size = (1,1), strides = (1,1),

                               padding = "same", kernel_regularizer = kernel_regularizer)

    skip_connection_2 = tf.add(deconv_layer_4_7, layer3_conv_1x1)

    

    output = tf.layers.conv2d_transpose(skip_connection_2, num_classes, 4, (2,2), padding="same", kernel_regularizer = kernel_regularizer)  

    return output
def optimize(nn_last_layer, correct_label, learning_rate):

    """

    Build the TensorFLow loss and optimizer operations.

    :param nn_last_layer: TF Tensor of the last layer in the neural network

    :param correct_label: TF Placeholder for the correct label image

    :param learning_rate: TF Placeholder for the learning rate

    :param num_classes: Number of classes to classify

    :return: Tuple of (logits, train_op, cross_entropy_loss)

    """

    logits = tf.reshape(nn_last_layer, (-1, args.NUM_CLASSES))

    labels = tf.reshape(correct_label, (-1, args.NUM_CLASSES))

    

    # Loss function

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    

    # Optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate)

    

    train_op = optimizer.minimize(cross_entropy_loss)

    

    return logits, train_op, cross_entropy_loss
def plot_loss(runs_dir, loss, folder_name):

    _, axes = plt.subplots()

    plt.plot(range(0, len(loss)), loss)

    plt.title('Cross-entropy loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.grid()

    if os.path.exists(runs_dir):

        shutil.rmtree(runs_dir)

    os.makedirs(runs_dir)



    output_file = os.path.join(runs_dir, folder_name + ".png")

    plt.savefig(output_file)
# Define batch function



def get_batches(image_paths, imseg_paths, batch_size):

    

    for batch in range(0, len(image_paths), batch_size):

        images = []

        imsegs = []

        

        for i in range(batch, batch + batch_size):

            image, imseg = read_image(image_paths[i], imseg_paths[i])

            

            images.append(image)

            imsegs.append(imseg)

            

        yield np.array(images), np.array(imsegs)    
def train_nn(sess, train_op, keep_prob, cross_entropy_loss, input_image, correct_label, learning_rate):

    """

    Train neural network and print out the loss during training.

    :param sess: TF Session

    :param epochs: Number of epochs

    :param batch_size: Batch size

    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)

    :param train_op: TF Operation to train the neural network

    :param cross_entropy_loss: TF Tensor for the amount of loss

    :param input_image: TF Placeholder for input images

    :param correct_label: TF Placeholder for label images

    :param keep_prob: TF Placeholder for dropout keep probability

    :param learning_rate: TF Placeholder for learning rate

    """

    print("Training started")

    lr = args.LEARNING_RATE

    

    sess.run(tf.global_variables_initializer())

    

    losses = []

    for epoch in range(args.EPOCHS):

        loss = None

        s_time = time.time()

        for images, labels in get_batches(X_train, y_train, args.BATCH_SIZE):

            _, loss = sess.run([train_op, cross_entropy_loss],

                                feed_dict={input_image: images,

                                           correct_label: labels,

                                           keep_prob: keep_prob,

                                           learning_rate: lr})

            losses.append(loss)

        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, epochs, loss, str(timedelta(seconds=(time.time() - s_time)))))

    plot_loss(RUNS_DIR, losses, "loss_graph")

    

    pass
def run():



    data_dir = './'

    runs_dir = './runs'

    height, width = args.IMAGE_SHAPE



    with tf.Session() as sess:

        # Path to vgg model

        vgg_path = os.path.join(data_dir, 'vgg')



        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        output_layer = layers(layer3, layer4, layer7, args.NUM_CLASSES)



        correct_label = tf.placeholder(tf.float32, shape=[None, None, None, None])

        learning_rate = tf.placeholder(tf.float32, shape=[])



        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate) 



        train_nn(sess, train_op, keep_prob, cross_entropy_loss, input_image, correct_label, learning_rate)





if __name__ == '__main__':

    run()
