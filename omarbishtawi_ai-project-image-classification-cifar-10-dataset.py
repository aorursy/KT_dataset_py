from urllib.request import urlretrieve

from os.path import isfile, isdir

from tqdm import tqdm 

import tarfile



cifar10_dataset_folder_path = '../input/cifar10/cifar-10-batches-py'



class DownloadProgress(tqdm):

    last_block = 0



    def hook(self, block_num=1, block_size=1, total_size=None):

        self.total = total_size

        self.update((block_num - self.last_block) * block_size)

        self.last_block = block_num



""" 

    check if the data (zip) file is already downloaded

    if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz

"""





if not isdir(cifar10_dataset_folder_path):

    with tarfile.open('cifar-10-batches-py') as tar:

        tar.extractall()

        tar.close()

import pickle

import numpy as np

import matplotlib.pyplot as plt
def load_label_names():

    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):

    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:

        # note the encoding type is 'latin1'

        batch = pickle.load(file, encoding='latin1')

        

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)

    labels = batch['labels']

        

    return features, labels
def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    

    if not (0 <= sample_id < len(features)):

        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))

        return None



    print('\nStats of batch #{}:'.format(batch_id))

    print('# of Samples: {}\n'.format(len(features)))

    

    label_names = load_label_names()

    label_counts = dict(zip(*np.unique(labels, return_counts=True)))

    for key, value in label_counts.items():

        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    

    sample_image = features[sample_id]

    sample_label = labels[sample_id]

    

    print('\nExample of Image {}:'.format(sample_id))

    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))

    print('Image - Shape: {}'.format(sample_image.shape))

    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

    

    plt.imshow(sample_image)
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np



# Explore the dataset

batch_id = 3

sample_id = 9999

display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
def normalize(x):

    """

        argument

            - x: input image data in numpy array [32, 32, 3]

        return

            - normalized x 

    """

    min_val = np.min(x)

    max_val = np.max(x)

    x = (x-min_val) / (max_val-min_val)

    return x
def one_hot_encode(x):

    """

        argument

            - x: a list of labels

        return

            - one hot encoding matrix (number of labels, number of class)

    """

    encoded = np.zeros((len(x), 10))

    

    for idx, val in enumerate(x):

        encoded[idx][val] = 1

    

    return encoded
def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):

    features = normalize(features)

    labels = one_hot_encode(labels)



    pickle.dump((features, labels), open(filename, 'wb'))





def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):

    n_batches = 5

    valid_features = []

    valid_labels = []



    for batch_i in range(1, n_batches + 1):

        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        

        # find index to be the point as validation data in the whole dataset of the batch (10%)

        index_of_validation = int(len(features) * 0.1)



        # preprocess the 90% of the whole dataset of the batch

        # - normalize the features

        # - one_hot_encode the lables

        # - save in a new file named, "preprocess_batch_" + batch_number

        # - each file for each batch

        _preprocess_and_save(normalize, one_hot_encode,

                             features[:-index_of_validation], labels[:-index_of_validation], 

                             'preprocess_batch_' + str(batch_i) + '.p')



        # unlike the training dataset, validation dataset will be added through all batch dataset

        # - take 10% of the whold dataset of the batch

        # - add them into a list of

        #   - valid_features

        #   - valid_labels

        valid_features.extend(features[-index_of_validation:])

        valid_labels.extend(labels[-index_of_validation:])



    # preprocess the all stacked validation dataset

    _preprocess_and_save(normalize, one_hot_encode,

                         np.array(valid_features), np.array(valid_labels),

                         'preprocess_validation.p')



    # load the test dataset

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:

        batch = pickle.load(file, encoding='latin1')



    # preprocess the testing data

    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)

    test_labels = batch['labels']



    # Preprocess and Save all testing data

    _preprocess_and_save(normalize, one_hot_encode,

                         np.array(test_features), np.array(test_labels),

                         'preprocess_training.p')
preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
import pickle



valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
!pip install tensorflow==1.5
# Remove previous weights, bias, inputs, etc..

import tensorflow as tf

from tensorflow.python.framework import ops

print(tf.__version__)

ops.reset_default_graph()



# Inputs

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')

y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



def conv_net(x, keep_prob):

    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))

    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))

    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))

    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))



    # 1, 2

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')

    conv1 = tf.nn.relu(conv1)

    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv1_bn = tf.layers.batch_normalization(conv1_pool)



    # 3, 4

    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')

    conv2 = tf.nn.relu(conv2)

    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    

    conv2_bn = tf.layers.batch_normalization(conv2_pool)

  

    # 5, 6

    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')

    conv3 = tf.nn.relu(conv3)

    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  

    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    

    # 7, 8

    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')

    conv4 = tf.nn.relu(conv4)

    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    

    # 9

    flat = tf.contrib.layers.flatten(conv4_bn)  



    # 10

    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)

    full1 = tf.nn.dropout(full1, keep_prob)

    full1 = tf.layers.batch_normalization(full1)

    

    # 11

    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)

    full2 = tf.nn.dropout(full2, keep_prob)

    full2 = tf.layers.batch_normalization(full2)

    

    # 12

    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)

    full3 = tf.nn.dropout(full3, keep_prob)

    full3 = tf.layers.batch_normalization(full3)    

    

    # 13

    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)

    full4 = tf.nn.dropout(full4, keep_prob)

    full4 = tf.layers.batch_normalization(full4)        

    

    # 14

    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)

    return out
epochs = 10

batch_size = 128

keep_probability = 0.7

learning_rate = 0.001
logits = conv_net(x, keep_prob)

model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training



# Loss and Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Accuracy

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):

    session.run(optimizer, 

                feed_dict={

                    x: feature_batch,

                    y: label_batch,

                    keep_prob: keep_probability

                })
def print_stats(session, feature_batch, label_batch, cost, accuracy):

    loss = sess.run(cost, 

                    feed_dict={

                        x: feature_batch,

                        y: label_batch,

                        keep_prob: 1.

                    })

    valid_acc = sess.run(accuracy, 

                         feed_dict={

                             x: valid_features,

                             y: valid_labels,

                             keep_prob: 1.

                         })

    

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
def batch_features_labels(features, labels, batch_size):

    """

    Split features and labels into batches

    """

    for start in range(0, len(features), batch_size):

        end = min(start + batch_size, len(features))

        yield features[start:end], labels[start:end]



def load_preprocess_training_batch(batch_id, batch_size):

    """

    Load the Preprocessed Training data and return them in batches of <batch_size> or less

    """

    filename = 'preprocess_batch_' + str(batch_id) + '.p'

    features, labels = pickle.load(open(filename, mode='rb'))



    # Return the training data in batches of size <batch_size> or less

    return batch_features_labels(features, labels, batch_size)
save_model_path = './image_classification'



print('Training...')

with tf.Session() as sess:

    # Initializing the variables

    sess.run(tf.global_variables_initializer())

    

    # Training cycle

    for epoch in range(epochs):

        # Loop over all batches

        n_batches = 5

        for batch_i in range(1, n_batches + 1):

            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):

                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

                

            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

            print_stats(sess, batch_features, batch_labels, cost, accuracy)

            

    # Save Model

    saver = tf.train.Saver()

    save_path = saver.save(sess, save_model_path)
import pickle

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer



def batch_features_labels(features, labels, batch_size):

    """

    Split features and labels into batches

    """

    for start in range(0, len(features), batch_size):

        end = min(start + batch_size, len(features))

        yield features[start:end], labels[start:end]



def display_image_predictions(features, labels, predictions, top_n_predictions):

    n_classes = 10

    label_names = load_label_names()

    label_binarizer = LabelBinarizer()

    label_binarizer.fit(range(n_classes))

    label_ids = label_binarizer.inverse_transform(np.array(labels))



    fig, axies = plt.subplots(nrows=top_n_predictions, ncols=2, figsize=(20, 10))

    fig.tight_layout()

    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)



    n_predictions = 3

    margin = 0.05

    ind = np.arange(n_predictions)

    width = (1. - 2. * margin) / n_predictions

   

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):

        if (image_i < top_n_predictions):

            pred_names = [label_names[pred_i] for pred_i in pred_indicies]

            correct_name = label_names[label_id]

            

            axies[image_i][0].imshow((feature*255).astype(np.int32, copy=False))

            axies[image_i][0].set_title(correct_name)

            axies[image_i][0].set_axis_off()



            axies[image_i][1].barh(ind + margin, pred_values[:3], width)

            axies[image_i][1].set_yticks(ind + margin)

            axies[image_i][1].set_yticklabels(pred_names[::-1])

            axies[image_i][1].set_xticks([0, 0.5, 1.0])
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import tensorflow as tf

import pickle

import random



save_model_path = './image_classification'

batch_size = 64

n_samples = 10

top_n_predictions = 5



def test_model():

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))

    loaded_graph = tf.Graph()



    with tf.Session(graph=loaded_graph) as sess:

        # Load model

        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        loader.restore(sess, save_model_path)



        # Get Tensors from loaded model

        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')

        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')

        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        

        # Get accuracy in batches for memory limitations

        test_batch_acc_total = 0

        test_batch_count = 0

        

        for train_feature_batch, train_label_batch in batch_features_labels(test_features, test_labels, batch_size):

            test_batch_acc_total += sess.run(

                loaded_acc,

                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})

            test_batch_count += 1



        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))



        # Print Random Samples

        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))

        random_test_predictions = sess.run(

            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),

            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})

        display_image_predictions(random_test_features, random_test_labels, random_test_predictions, top_n_predictions)





test_model()