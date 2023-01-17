import pickle

import os



import numpy as np



from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.utils import shuffle



from tqdm import tqdm



from tensorflow.keras.preprocessing.sequence import pad_sequences





UNKNOWN = 1000000



class DataHelper:



    def __init__(self, num_words_to_keep):

        self.label_encoder = LabelEncoder()

        self.one_hot_encoder = OneHotEncoder(sparse=False)

        self.tokenizer = Tokenizer(num_words_to_keep)



    def train_tokenizer(self, text):

        self.tokenizer.fit_on_texts(text)



    def pickle_everything_to_disk(self, dir):

        with open(os.path.join(dir, 'label_encoder.pickle'), 'wb') as handle:

            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(dir, 'one_hot_encoder.pickle'), 'wb') as handle:

            pickle.dump(self.one_hot_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(dir, 'tokenizer.pickle'), 'wb') as handle:

            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



    @staticmethod

    def load_all_pickles(dir):

        with open(os.path.join(dir, 'label_encoder.pickle'), 'rb') as handle:

            label_encoder = pickle.load(handle)

        with open(os.path.join(dir, 'one_hot_encoder.pickle'), 'rb') as handle:

            one_hot_encoder = pickle.load(handle)

        with open(os.path.join(dir, 'tokenizer.pickle'), 'rb') as handle:

            tokenizer = pickle.load(handle)

        return label_encoder, one_hot_encoder, tokenizer



    def convert_to_indices(self, text):

        return self.tokenizer.texts_to_sequences(text)



    def train_one_hot_encoder(self, train_labels):

        integer_encoded = self.label_encoder.fit_transform(train_labels)

        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        integer_encoded = integer_encoded.tolist()

        integer_encoded.append([UNKNOWN])

        self.one_hot_encoder.fit_transform(integer_encoded)

        return self.one_hot_encoder, self.label_encoder



    def encode_to_one_hot(self, labels_list):

        # sklearn.LabelEncoder with never seen before values

        le_dict = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))

        integer_encoded = [le_dict.get(i, UNKNOWN) for i in labels_list]

        integer_encoded = np.array(integer_encoded)

        #integer_encoded = self.label_encoder.transform(labels_list)

        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)



        return self.one_hot_encoder.transform(integer_encoded)



    def prepare_for_tensorflow(self, text_list, labels_list, path_list, num_words_x_doc):

        text_list = [element.decode('UTF-8') for element in text_list]

        labels_list = [element.decode('UTF-8') for element in labels_list]

        path_list = [element.decode('UTF-8') for element in path_list]



        text_indices = self.tokenizer.texts_to_sequences(text_list)



        text_indices = pad_sequences(text_indices, maxlen=num_words_x_doc, value=0.)

        integer_encoded = self.label_encoder.transform(labels_list)

        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        y_one_hot = self.one_hot_encoder.transform(integer_encoded)



        return text_indices, y_one_hot, path_list



    def load_shuffle_data(self, train_path, val_path):

        print("Loading data...")



        text_train, label_train, img_train = self.load_data(train_path)

        text_val, label_val, img_val = self.load_data(val_path)



        text_train, label_train, img_train = shuffle(text_train, label_train, img_train, random_state=10)



        print("Train/Dev split: {:d}/{:d}".format(len(text_train), len(text_val)))

        return text_train, label_train, img_train, text_val, label_val, img_val



    @staticmethod

    def load_data(train_file):

        text = []

        label = []

        img = []

        with open(train_file) as tr:

            for line in tqdm(tr.readlines()):

                line = line.replace("\n", "")

                line = line.split('|')

                text.append(line[0])

                label.append(line[1])

                img.append(line[2])

        return text, label, img
num_words_to_keep = 30000

num_words_x_doc = 100

train_path="/kaggle/input/rectangle-ellipse-multimodal/rectangle_ellipse_multimodal/train.csv" # csv file containing text|class|image_path

val_path="/kaggle/input/rectangle-ellipse-multimodal/rectangle_ellipse_multimodal/val.csv" # csv file containing text|class|image_path

save_model_dir_name="model" # dir used to save the model

abs_dirname = os.path.dirname(os.path.abspath(train_path))

print("Absolute dir name:", abs_dirname)

        

data_helper = DataHelper(num_words_to_keep)



train_x, train_y, train_img_paths, test_x, test_y, test_imgs_paths = data_helper.load_shuffle_data(train_path, val_path)



data_helper.train_one_hot_encoder(train_y)

train_y = data_helper.encode_to_one_hot(train_y)

test_y = data_helper.encode_to_one_hot(test_y)



data_helper.train_tokenizer(train_x)



train_x = data_helper.convert_to_indices(train_x)

test_x = data_helper.convert_to_indices(test_x)



train_x = pad_sequences(train_x, maxlen=num_words_x_doc, value=0.)

test_x = pad_sequences(test_x, maxlen=num_words_x_doc, value=0.)
if not os.path.exists(save_model_dir_name):

    os.makedirs(save_model_dir_name)

    

data_helper.pickle_everything_to_disk(save_model_dir_name)
import os



for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



class batch_norm(object):

    """Code modification of http://stackoverflow.com/a/33950177"""



    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):

        with tf.variable_scope(name):

            self.epsilon = epsilon

            self.momentum = momentum



            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

            self.name = name



    def __call__(self, x, train=True):

        shape = x.get_shape().as_list()



        if train:

            with tf.variable_scope(self.name) as scope:

                self.beta = tf.get_variable("beta", [shape[-1]],

                                            initializer=tf.constant_initializer(0.))

                self.gamma = tf.get_variable("gamma", [shape[-1]],

                                             initializer=tf.random_normal_initializer(1., 0.02))



                try:

                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

                except:

                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')



                ema_apply_op = self.ema.apply([batch_mean, batch_var])

                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)



                with tf.control_dependencies([ema_apply_op]):

                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)

        else:

            mean, var = self.ema_mean, self.ema_var



        normed = tf.nn.batch_norm_with_global_normalization(

            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)



        return normed



    

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    shape = input_.get_shape().as_list()



    with tf.variable_scope(scope or "Linear"):

        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,

                                 tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size],

                               initializer=tf.constant_initializer(bias_start))

        if with_w:

            return tf.matmul(input_, matrix) + bias, matrix, bias

        else:

            return tf.matmul(input_, matrix) + bias

        

def conv2d(input_, output_dim,

           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,

           name="conv2d"):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],

                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')



        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.bias_add(conv, biases)

        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())



        return conv



def deconv2d(input_, output_shape,

             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,

             name="deconv2d", with_w=False):

    with tf.variable_scope(name):

        # tmp_tensor = tf.placeholder(dtype=tf.float32, shape=output_shape)

        # deconv_shape = tf.stack(output_shape)



        # filter : [height, width, output_channels, in_channels]

        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],

                            initializer=tf.random_normal_initializer(stddev=stddev))



        try:

            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

            # deconv = tf.nn.conv2d_transpose(input_, w, output_shape=deconv_shape, strides=[1, d_h, d_w, 1])



        # Support for verisons of TensorFlow before 0.7.0

        except AttributeError:

            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

            # deconv = tf.nn.deconv2d(input_, w, output_shape=deconv_shape, strides=[1, d_h, d_w, 1])



        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases),  tf.shape(deconv))



        if with_w:

            return deconv, w, biases

        else:

            return deconv

        

def lrelu(x, leak=0.2, name="lrelu"):

    return tf.maximum(x, leak * x)        


from math import ceil





class TextImgCNN(object):

    """

    A CNN for text classification.

    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    """



    def __init__(self, sequence_length, num_classes, vocab_size,

                 embedding_size, filter_sizes, num_filters, output_image_width, encoding_height,

                 l2_reg_lambda=0.5, batch_size=32):



        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

        self.input_image = tf.placeholder(tf.float32, [None, output_image_width, output_image_width, 3], name="input_mask")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.g_bn0 = batch_norm(name='generator_bn0')

        self.g_bn1 = batch_norm(name='generator_bn1')

        self.g_bn2 = batch_norm(name='generator_bn2')

        self.g_bn3 = batch_norm(name='genrerator_bn3')

        self.d_bn1 = batch_norm(name='discriminator_bn1')

        self.d_bn2 = batch_norm(name='discriminator_bn2')

        self.d_bn3 = batch_norm(name='discriminator_bn3')

        self.d_bn4 = batch_norm(name='discriminator_bn4')



        # Embedding layer

        with tf.name_scope("generator_embedding"):

            self.W = tf.Variable(

                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),

                name="W")

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



        # Create a convolution + maxpool layer for each filter size

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):

            with tf.name_scope("generator_conv-maxpool-%s" % filter_size):

                # Convolution Layer

                filter_shape = [filter_size, embedding_size, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(

                    self.embedded_chars_expanded,

                    W,

                    strides=[1, 1, 1, 1],

                    padding="VALID",

                    name="conv")

                # Apply nonlinearity

                h = tf.nn.relu(tf.nn.bias_add(conv, b))

                # Maxpooling over the outputs

                pooled = tf.nn.max_pool(

                    h,

                    ksize=[1, sequence_length - filter_size + 1, 1, 1],

                    strides=[1, 1, 1, 1],

                    padding='VALID',

                    name="pool")

                pooled_outputs.append(pooled)



        # Combine all the pooled features

        num_filters_total = num_filters * len(filter_sizes)

        self.h_pool = tf.concat(pooled_outputs, 3)



        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])



        # Deconv from https://github.com/paarthneekhara/text-to-image/blob/master/model.py

        gf_dim = 64 # Number of conv

        s = output_image_width # src image size

        s2, s4, s8, s16 = int(ceil(s / 2)), int(ceil(s / 4)), int(ceil(s / 8)), int(ceil(s / 16))



        fc0 = linear(self.h_pool_flat, gf_dim * 8 * s16 * s16, 'generator_h0_lin')



        h0 = tf.reshape(fc0, [-1, s16, s16, gf_dim * 8])

        h0 = tf.nn.relu(self.g_bn0(h0))



        h1 = deconv2d(h0, [tf.shape(h0)[0], s8, s8, gf_dim * 4], name='generator_h1')

        h1 = tf.nn.relu(self.g_bn1(h1))



        h2 = deconv2d(h1, [tf.shape(h0)[0], s4, s4, gf_dim * 2], name='generator_h2')

        h2 = tf.nn.relu(self.g_bn2(h2))



        h3 = deconv2d(h2, [tf.shape(h0)[0], s2, s2, gf_dim * 1], name='generator_h3')

        h3 = tf.nn.relu(self.g_bn3(h3))



        self.reshaped_text_features = deconv2d(h3, [tf.shape(h0)[0], s, s, 3], name='generator_encoded_text')

        # self.reshaped_text_features = tf.tanh(self.reshaped_text_features) / 2. + 0.5

        self.reshaped_text_features = tf.sigmoid(self.reshaped_text_features, name='generator_encoded_text_sigmoid')



        sh0, sh1, sh2, sh3 = int(ceil(s / 2)), int(ceil(s / 4)), int(ceil(s / 8)), int(ceil(s / 16))



        hh0 = lrelu(conv2d(self.reshaped_text_features, gf_dim, name='discriminator_h0_conv'))

        hh1 = lrelu(self.d_bn1(conv2d(hh0, gf_dim / 2, name='discriminator_h1_conv')))

        hh2 = lrelu(self.d_bn2(conv2d(hh1, gf_dim / 4, name='discriminator_h2_conv')))

        hh3 = lrelu(self.d_bn3(conv2d(hh2, gf_dim / 8, name='discriminator_h3_conv')))



        hh3_flat_size = int(sh3*sh3*(gf_dim / 8)) 

        self.h_pool_flat_2 = tf.reshape(hh3, [-1, hh3_flat_size])



        fc1_size = 1024 # hh3_flat_size/2

        fc1 = linear(self.h_pool_flat_2, fc1_size, 'discriminator_f1_lin')



        fc2_size = 512  # hh3_flat_size/2

        fc2 = linear(fc1, fc2_size, 'discriminator_f2_lin')



        # Add dropout

        with tf.name_scope("dropout"):

            self.h_drop = tf.nn.dropout(fc2, self.dropout_keep_prob)



        # Final (unnormalized) scores and predictions

        with tf.name_scope("discriminator_output"):

            W = tf.get_variable(

                "W",

                shape=[fc2_size, num_classes],

                initializer=tf.glorot_uniform_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

            self.predictions = tf.argmax(self.scores, 1, name="predictions")



        # Calculate mean cross-entropy loss

        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.loss_d = tf.reduce_mean(losses)

            self.loss_g = tf.reduce_sum(tf.square(tf.subtract(self.reshaped_text_features, self.input_image)))

            self.loss_full = self.loss_d + l2_reg_lambda * self.loss_g



        # Accuracy

        with tf.name_scope("accuracy"):

            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

batch_size_test = 32

embedding_dim = 128 # Dimensionality of character embedding

filter_sizes = "3,4,5" # Comma-separated filter sizes (default: '3,4,5')

num_filters = 128 # Number of filters per filter size

dropout_keep_prob = 0.5 # Dropout keep probability

l2_reg_lambda = 1.0 # L2 regularization lambda

batch_size = 32 # Batch Size

num_epochs=3 # Number of training epochs 

evaluate_every=1000 # Evaluate model on dev set after this many steps 

num_checkpoints=1 # Number of checkpoints to store 

patience=5 # Stop criteria 

output_image_width=100 # Size of output Image plus embedding 



encoding_height=10 # Height of the output embedding 
import datetime

#import os

import time

import cv2



def train(x_train, y_train, img_train, x_test, y_test, img_test, words_to_keep, output_image_width, encoding_height,

          patience_init_val):

    def train_step(x_batch, y_batch, images_batch):

        """

        A single training step

        """

        feed_dict = {

            cnn.input_x: x_batch,

            cnn.input_y: y_batch,

            cnn.input_image: images_batch,

            cnn.dropout_keep_prob: dropout_keep_prob

        }



        _, step, summaries, loss_g, loss_d, loss_full, accuracy = sess.run(

            [train_op_full, global_step, train_summary_op, cnn.loss_g, cnn.loss_d, cnn.loss_full, cnn.accuracy], feed_dict)



        time_str = datetime.datetime.now().isoformat()

        #print("TRAIN: {}: step {}, loss_d {:g}, loss_g {:g}, loss_full {:g}, acc {:g}".format(time_str, step, loss_d, loss_g, loss_full, accuracy))

        train_summary_writer.add_summary(summaries, step)



    def dev_step_only_accuracy(x_batch, y_batch, images_batch):

        """

        Evaluates model on a dev set

        """

        feed_dict = {

            cnn.input_x: x_batch,

            cnn.input_y: y_batch,

            cnn.input_image: images_batch,

            cnn.dropout_keep_prob: 1.0

        }

        step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss_d, cnn.accuracy], feed_dict)

        return accuracy



    best_accuracy = 0

    patience = patience_init_val



    with tf.Graph().as_default():

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        with sess.as_default():



            cnn = TextImgCNN(

                sequence_length=x_train.shape[1],

                num_classes=y_train.shape[1],

                vocab_size=words_to_keep,

                embedding_size=embedding_dim,

                filter_sizes=list(map(int, filter_sizes.split(","))),

                num_filters=num_filters,

                output_image_width=output_image_width,

                encoding_height=encoding_height,

                l2_reg_lambda=l2_reg_lambda,

                batch_size=batch_size)



            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, img_train))

            dataset = dataset.batch(batch_size)

            train_iterator = dataset.make_initializable_iterator()

            sess.run(train_iterator.initializer)

            next_element = train_iterator.get_next()



            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, img_test))

            test_dataset = test_dataset.batch(batch_size_test)

            test_iterator = test_dataset.make_initializable_iterator()

            next_test_element = test_iterator.get_next()



            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'discriminator' in var.name]

            g_vars = [var for var in t_vars if 'generator' in var.name]



            global_step = tf.Variable(0, name="global_step", trainable=False)

            train_op_full = tf.train.AdamOptimizer(1e-2).minimize(cnn.loss_full, global_step=global_step)



            # Output directory for models and summaries

            timestamp = str(int(time.time()))

            out_dir = save_model_dir_name

            if not os.path.exists(out_dir):

                os.makedirs(out_dir)



            print("Writing to {}\n".format(out_dir))



            # Summaries for loss and test_accuracy

            loss_summary = tf.summary.scalar("loss", cnn.loss_d)

            acc_summary = tf.summary.scalar("test_accuracy", cnn.accuracy)



            # Train Summaries

            train_summary_op = tf.summary.merge([loss_summary, acc_summary]) #, grad_summaries_merged])

            train_summary_dir = os.path.join(out_dir, "summaries", "train")

            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)



            # Dev summaries

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])



            # initialize all of the variables in the session

            sess.run(tf.global_variables_initializer())



            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):

                os.makedirs(checkpoint_dir)

                print("Creating new checkpoint dir", checkpoint_dir)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            else:

                # Load the saved meta graph and restore variables

                tf.initialize_all_variables().run()

                saver = tf.train.Saver(tf.all_variables())

                # restore gen_model

                saver.restore(sess, checkpoint_file)

                print("Restoring model from", checkpoint_dir)



            with open(os.path.join(out_dir, "results.txt"), "a") as resfile:

                resfile.write("Model dir: {}\n".format(out_dir))

                resfile.write("Dataset: {}\n".format(img_train[0]))





            train_length = len(x_train)

            val_length = len(x_test)



            for ep in range(num_epochs):

                print("***** Epoch " + str(ep) + " *****")

                sess.run(train_iterator.initializer)



                for b in range((train_length // batch_size) + 1):

                    images_batch = []

                    element = sess.run(next_element)



                    path_list = [el.decode('UTF-8') for el in element[2]]



                    for path in path_list:

                        path = os.path.join(abs_dirname, path)

                        #print("image path: " + path)

                        img = cv2.imread(path)

                        img = cv2.resize(img, (output_image_width, output_image_width))

                        img = img / 255

                        images_batch.append(img)



                    train_step(element[0], element[1], images_batch)



                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % evaluate_every == 0:

                        sess.run(test_iterator.initializer)



                    if current_step % evaluate_every == 0:

                        print("\nEvaluation:")

                        # Run one pass over the validation dataset.

                        sess.run(test_iterator.initializer)

                        correct = 0

                        for b in range((val_length // batch_size_test) + 1):

                            test_img_batch = []

                            test_element = sess.run(next_test_element)



                            test_path_list = [el.decode('UTF-8') for el in test_element[2]]



                            for path in test_path_list:

                                path = os.path.join(abs_dirname, path)

                                img = cv2.imread(path)

                                img = cv2.resize(img, (output_image_width, output_image_width))

                                img = img / 255

                                test_img_batch.append(img)



                            acc = dev_step_only_accuracy(test_element[0], test_element[1], test_img_batch)

                            correct += acc * len(test_path_list)

                            

                        test_accuracy = correct / val_length



                        if test_accuracy > best_accuracy:

                            best_accuracy = test_accuracy

                            patience = patience_init_val

                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)

                            print("Saved model checkpoint to {}\n".format(path))

                        else:

                            patience -= 1



                        print("TEST: epoch: %d, step: %d, test acc: %f, best acc: %f, patience: %d\n" % (

                                ep, current_step, test_accuracy, best_accuracy, patience))



                        #with open(os.path.join(out_dir, "results.txt"), "a") as resfile:

                        #    resfile.write("epoch: %d, step: %d, test acc: %f, best acc: %f, patience: %d\n" % (

                        #        ep, current_step, test_accuracy, best_accuracy, patience))



                        if patience == 0:

                            return cnn

            return cnn
trained_cnn = train(train_x, train_y, train_img_paths, test_x, test_y, test_imgs_paths, num_words_to_keep, output_image_width, encoding_height, patience)
val_length = len(test_x)



with tf.Graph().as_default():

    sess = tf.Session()

    with sess.as_default():

        val_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_imgs_paths))

        val_dataset = val_dataset.batch(batch_size)

        val_iterator = val_dataset.make_initializable_iterator()

        val_next_element = val_iterator.get_next()



        sess.run(val_iterator.initializer)



        print("Creating VAL images from CNN features...")

        correct = 0

        for b in tqdm(range((val_length // batch_size) + 1)):

            images = []

            element = sess.run(val_next_element)

            path_list = [el.decode('UTF-8') for el in element[2]]



            feed_dict = {

                trained_cnn.input_x: element[0],

                trained_cnn.input_y: element[1],

                trained_cnn.dropout_keep_prob: 1.0

            }

            acc, encoded_text_features, preds = sess.run([trained_cnn.accuracy, trained_cnn.reshaped_text_features, trained_cnn.predictions], feed_dict)

            pred_labels = list(label_encoder.inverse_transform(preds))

            correct += acc * len(path_list)  # batch_size



            # for features, src_img in zip(encoded_text_features, images):

            #     cv2.imshow("image_features", features)

            #     print("min:", features.min(), "max:", features.max())

            #     cv2.imshow("image_src", src_img)

            #     cv2.waitKey(0)



        test_accuracy = correct / val_length

        print("Test accuracy: {}/{}={}".format(correct, val_length, test_accuracy))