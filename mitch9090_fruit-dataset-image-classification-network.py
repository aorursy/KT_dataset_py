import tensorflow as tf

import numpy as np

import time

import os



IMAGE_HEIGHT = 100

IMAGE_WIDTH = 100

IMAGE_CHANNELS = 3

NETWORK_DEPTH = 4



data_dir = os.getcwd() + "/../input/" 

data_dir += os.listdir(data_dir)[0] + "/fruits-360/"

train_dir = data_dir + "Training/"

validation_dir = data_dir + "Test/"



batch_size = 60

input_size = IMAGE_HEIGHT * IMAGE_WIDTH * NETWORK_DEPTH

num_classes = len(os.listdir(train_dir))

# probability to keep the values after a training iteration

dropout = 0.8



initial_learning_rate = 0.001

final_learning_rate = 0.00001

learning_rate = initial_learning_rate



# number of iterations to run the training

iterations = 75000

# number of iterations after we display the loss and accuracy

acc_display_interval = 1000

# default number of iterations after we save the model

save_interval = 1000

step_display_interval = 100

# use the saved model and continue training

useCkpt = False

# placeholder for probability to keep the network parameters after an iteration

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# -------------------- Write/Read TF record logic --------------------

class ImageCoder(object):

    """Helper class that provides TensorFlow image coding utilities."""



    def __init__(self):

        # Create a single Session to run all image coding calls.

        self._sess = tf.Session()



        # Initializes function that decodes RGB JPEG data.

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)

        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)



    def decode_jpeg(self, image_data):

        image = self._sess.run(self._decode_jpeg,

                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3

        assert image.shape[2] == 3

        return image



def write_image_data(dir_name, tfrecords_name):

    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    coder = ImageCoder()

    image_count = 0

    index = -1

    classes_dict = {}



    for folder_name in os.listdir(dir_name):

        class_path = dir_name + '/' + folder_name + '/'

        index += 1

        classes_dict[index] = folder_name

        for image_name in os.listdir(class_path):

            image_path = class_path + image_name

            image_count += 1

            with tf.gfile.FastGFile(image_path, 'rb') as f:

                image_data = f.read()

                example = tf.train.Example(

                    features = tf.train.Features(

                        feature = {

                            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),

                            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image_data)]))

                        }

                    )

                )

                writer.write(example.SerializeToString())

    writer.close()

    print(classes_dict)

    return image_count, classes_dict



def parse_single_example(serialized_example):

    features = tf.parse_single_example(

        serialized_example,

        features={

            'image_raw': tf.FixedLenFeature([], tf.string),

            'label': tf.FixedLenFeature([], tf.int64)

        }

    )

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)

    image = tf.reshape(image, [100, 100, 3])

    label = tf.cast(features['label'], tf.int32)

    return image, label

def conv(input_tensor, name, kernel_width, kernel_height, num_out_activation_maps, stride_horizontal=1, stride_vertical=1, activation_fn=tf.nn.relu):

    prev_layer_output = input_tensor.get_shape()[-1].value

    with tf.variable_scope(name):

        weights = tf.get_variable('weights', [kernel_height, kernel_width, prev_layer_output, num_out_activation_maps], tf.float32,

                                  tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        biases = tf.get_variable("bias", [num_out_activation_maps], tf.float32, tf.constant_initializer(0.0))

        conv_layer = tf.nn.conv2d(input_tensor, weights, (1, stride_horizontal, stride_vertical, 1), padding='SAME')

        activation = activation_fn(tf.nn.bias_add(conv_layer, biases))

        return activation



def fully_connected(input_tensor, name, output_neurons, activation_fn=tf.nn.relu):

    n_in = input_tensor.get_shape()[-1].value

    with tf.variable_scope(name):

        weights = tf.get_variable('weights', [n_in, output_neurons], tf.float32,

                                  initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))

        biases = tf.get_variable("bias", [output_neurons], tf.float32, tf.constant_initializer(0.0))

        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)

        if activation_fn is None:

            return logits

        return activation_fn(logits)



def max_pool(input_tensor, name, kernel_height, kernel_width, stride_horizontal, stride_vertical):

    return tf.nn.max_pool(input_tensor,

                          ksize=[1, kernel_height, kernel_width, 1],

                          strides=[1, stride_horizontal, stride_vertical, 1],

                          padding='VALID',

                          name=name)



def loss(logits, onehot_labels):

    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels, name='xentropy')

    loss = tf.reduce_mean(xentropy, name='loss')

    return loss



def update_learning_rate(acc, learn_rate):

    return learn_rate - acc * learn_rate * 0.9
# perform data augmentation on images

# add random hue and saturation

# randomly flip the image vertically and horizontally

def augment_image(image):

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.random_hue(image, 0.02)

    image = tf.image.random_saturation(image, 0.9, 1.2)

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    return build_hsv_grayscale_image(image)



# converts the image from RGB to HSV and

# adds a 4th channel to the HSV ones that contains the image in gray scale

# for test just convert the image to HSV and add the gray scale channel

def build_hsv_grayscale_image(image):

    image = tf.image.convert_image_dtype(image, tf.float32)

    gray_image = tf.image.rgb_to_grayscale(image)

    image = tf.image.rgb_to_hsv(image)

    rez = tf.concat([image, gray_image], 2)

    return rez
def build_datasets(train_file, test_file, batch_size):

    train_dataset = tf.data.TFRecordDataset(train_file).repeat()

    train_dataset = train_dataset.map(parse_single_example).map(lambda image, label: (augment_image(image), label))

    train_dataset = train_dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    train_dataset = train_dataset.batch(batch_size)

    validation_dataset = tf.data.TFRecordDataset(train_file)

    validation_dataset = validation_dataset.map(parse_single_example).map(lambda image, label: (build_hsv_grayscale_image(image), label))

    validation_dataset = validation_dataset.batch(batch_size)

    test_dataset = tf.data.TFRecordDataset(test_file)

    test_dataset = test_dataset.map(parse_single_example).map(lambda image, label: (build_hsv_grayscale_image(image), label))

    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, validation_dataset, test_dataset



def train_model(session, train_operation, loss_operation, correct_prediction, iterator_map):

    global learning_rate

    time1 = time.time()

    train_iterator = iterator_map["train_iterator"]

    validation_iterator = iterator_map["validation_iterator"]

    validation_init_op = iterator_map["validation_init_op"]

    train_images_with_labels = train_iterator.get_next()

    validation_images_with_labels = validation_iterator.get_next()

    for i in range(1, iterations + 1):

        batch_x, batch_y = session.run(train_images_with_labels)

        batch_x = np.reshape(batch_x, [batch_size, input_size])

        session.run(train_operation, feed_dict={X: batch_x, Y: batch_y})



        if i % step_display_interval == 0:

            time2 = time.time()

            print("time: %.4f step: %d" % (time2 - time1, i))

            time1 = time.time()



        if i % acc_display_interval == 0:

            acc_value, loss = calculate_intermediate_accuracy_and_loss(session, correct_prediction, loss_operation,

                                                                       validation_images_with_labels, validation_init_op, train_images_count)

            learning_rate = update_learning_rate(acc_value, learn_rate=learning_rate)

            print("step: %d loss: %.4f accuracy: %.4f" % (i, loss, acc_value))

        if i % save_interval == 0:

            # save the weights and the meta data for the graph

            saver.save(session, './model.ckpt')

            tf.train.write_graph(session.graph_def,'./', 'graph.pbtxt')





def calculate_intermediate_accuracy_and_loss(session, correct_prediction, loss_operation, test_images_with_labels, test_init_op, total_image_count):

    sess.run(test_init_op)

    loss = 0

    predicted = 0

    count = 0

    total = 0

    while total < total_image_count:

        test_batch_x, test_batch_y = session.run(test_images_with_labels)

        test_batch_x = np.reshape(test_batch_x, [-1, input_size])

        l, p = session.run([loss_operation, correct_prediction], feed_dict={X: test_batch_x, Y: test_batch_y})

        loss += l

        predicted += np.sum(p)

        count += 1

        total += len(p)

    return predicted / total_image_count, loss / count





def test_model(sess, pred, iterator, total_images, file_name):

    images_left_to_process = total_images

    total_number_of_images = total_images

    images_with_labels = iterator.get_next()

    correct = 0

    while images_left_to_process > 0:

        batch_x, batch_y = sess.run(images_with_labels)

        batch_x = np.reshape(batch_x, [-1, input_size])

        # the results of the classification is an array of 1 and 0, 1 is a correct classification

        results = sess.run(pred, feed_dict={X: batch_x, Y: batch_y})

        images_left_to_process = images_left_to_process - len(results)

        correct = correct + np.sum(results)

        print("Predicted %d out of %d; partial accuracy %.4f" % (correct, total_number_of_images - images_left_to_process,

                                                                 correct / (total_number_of_images - images_left_to_process)))

    print("Final accuracy on %s data: %.8f" % (file_name, correct / total_number_of_images))
def conv_net(input_layer, dropout):

    # number of activation maps for each convolutional layer

    number_of_act_maps_conv1 = 16

    number_of_act_maps_conv2 = 32

    number_of_act_maps_conv3 = 64

    number_of_act_maps_conv4 = 128



    # number of outputs for each fully connected layer

    number_of_fcl_outputs1 = 1024

    number_of_fcl_outputs2 = 256



    input_layer = tf.reshape(input_layer, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, NETWORK_DEPTH])



    conv1 = conv(input_layer, 'conv1', kernel_width=5, kernel_height=5, num_out_activation_maps=number_of_act_maps_conv1)

    conv1 = max_pool(conv1, 'max_pool1', kernel_height=2, kernel_width=2, stride_horizontal=2, stride_vertical=2)



    conv2 = conv(conv1, 'conv2', kernel_width=5, kernel_height=5, num_out_activation_maps=number_of_act_maps_conv2)

    conv2 = max_pool(conv2, 'max_pool2', kernel_height=2, kernel_width=2, stride_horizontal=2, stride_vertical=2)



    conv3 = conv(conv2, 'conv3', kernel_width=5, kernel_height=5, num_out_activation_maps=number_of_act_maps_conv3)

    conv3 = max_pool(conv3, 'max_pool3', kernel_height=2, kernel_width=2, stride_horizontal=2, stride_vertical=2)



    conv4 = conv(conv3, 'conv4', kernel_width=5, kernel_height=5, num_out_activation_maps=number_of_act_maps_conv4)

    conv4 = max_pool(conv4, 'max_pool4', kernel_height=2, kernel_width=2, stride_horizontal=2, stride_vertical=2)



    flattened_shape = np.prod([s.value for s in conv4.get_shape()[1:]])

    net = tf.reshape(conv4, [-1, flattened_shape], name="flatten")



    fcl1 = fully_connected(net, 'fcl1', number_of_fcl_outputs1)

    fcl1 = tf.nn.dropout(fcl1, rate=1 - dropout)



    fcl2 = fully_connected(fcl1, 'fcl2', number_of_fcl_outputs2)

    fcl2 = tf.nn.dropout(fcl2, rate=1 - dropout)



    out = fully_connected(fcl2, 'out', num_classes, activation_fn=None)



    return out
# ------------------------------------------------------------

train_images_count, fruit_labels = write_image_data(train_dir, "train.tfrecord")

test_images_count, _ = write_image_data(validation_dir, "test.tfrecord")

# ------------------------------------------------------------



with tf.Session() as sess:

    # placeholder for input layer

    X = tf.placeholder(tf.float32, [None, input_size], name="X")

    # placeholder for actual labels

    Y = tf.placeholder(tf.int64, [None], name="Y")

    

    # build the network

    logits = conv_net(input_layer=X, dropout=dropout)

    # apply softmax on the final layer

    prediction = tf.nn.softmax(logits)

    

    # calculate the loss using the predicted labels vs the expected labels

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    # use adaptive moment estimation optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss=loss)

    

    # calculate the accuracy for this training step

    correct_prediction = tf.equal(tf.argmax(prediction, 1), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    # input tfrecord file

    train_file = "train.tfrecord"

    test_file = "test.tfrecord"

    train_dataset, validation_dataset, test_dataset = build_datasets(train_file, test_file, batch_size)

    

    train_iterator = train_dataset.make_one_shot_iterator()

    validation_iterator = tf.data.Iterator.from_structure(validation_dataset.output_types, validation_dataset.output_shapes)

    validation_init_op = validation_iterator.make_initializer(validation_dataset)

    iterator_map = {"train_iterator": train_iterator,

                    "validation_iterator": validation_iterator,

                    "validation_init_op": validation_init_op}

    test_iterator = test_dataset.make_one_shot_iterator()



    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess.run(init)

    # restore the previously saved value if we wish to continue the training

    if useCkpt:

        ckpt = tf.train.get_checkpoint_state(".")

        saver.restore(sess, ckpt.model_checkpoint_path)



    train_model(sess, train_op, loss, correct_prediction, iterator_map)

    

    test_model(sess, correct_prediction, test_iterator, test_images_count, test_file)

    

    sess.close()