import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.display import display, Image, HTML

import os

import cv2



TRAIN_DIR = '../input/train'

TEST_DIR = '../input/test'

IMG_SIZE = 64
def create_input_data(im):

    img = cv2.imread(im, cv2.IMREAD_COLOR)

    if (img.shape[0] >= img.shape[1]): # height is greater than width

       resizeto = (IMG_SIZE, int (round (IMG_SIZE * (float (img.shape[1])  / img.shape[0]))));

    else:

       resizeto = (int (round (IMG_SIZE * (float (img.shape[0])  / img.shape[1]))), IMG_SIZE);

    

    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)

    img3 = cv2.copyMakeBorder(img2, 0, IMG_SIZE - img2.shape[0], 0, IMG_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

        

    return img3[:,:,::-1]

#     img = cv2.imread(im, cv2.IMREAD_COLOR)

#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

#     return np.array(img / 255)
def one_hot_encode(img):        

    if 'cat' in img:

        return np.array([0, 1])

    else:

        return np.array([1, 0])
training_img = []

training_label = []



testing_img = []

testing_label = []



for img in tqdm(os.listdir(TRAIN_DIR)):

    training_path = os.path.join(TRAIN_DIR, img)

    

    train_img = create_input_data(training_path)

    training_img.append(np.array(train_img))

    

    train_label = one_hot_encode(img)

    training_label.append(np.array(train_label))

    

for img in tqdm(os.listdir(TEST_DIR)):

    testing_path = os.path.join(TEST_DIR, img)

    

    test_img = create_input_data(testing_path)

    testing_img.append(np.array(test_img))

    

    test_label = one_hot_encode(img)

    testing_label.append(np.array(test_label))



training_img = np.array(training_img, dtype=np.float32)

training_label = np.array(training_label, dtype=np.float32)



testing_img = np.array(testing_img, dtype=np.float32)

testing_label = np.array(testing_label, dtype=np.float32)
index = 15000

plt.imshow(training_img[index])

plt.show()

print(training_label[index])
def neural_net_image_input(image_size):

    return tf.placeholder(tf.float32, [None] + list(image_size), 'x')



def neural_net_label_input():

    return tf.placeholder(tf.float32, [None, 2], 'y')



def neural_net_keep_prob():

    return tf.placeholder(tf.float32, None, 'keep_prob')
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize=(2,2), conv_strides=[1,1,1,1], pool_ksize=[1,2,2,1], pool_strides=[1,2,2,1]):

    dimension = x_tensor.get_shape().as_list()

    shape = list(conv_ksize + (dimension[-1],) + (conv_num_outputs,))

    weight = tf.Variable(tf.truncated_normal(shape, 0, 0.1))

    bias = tf.Variable(tf.zeros(conv_num_outputs))

    

    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=conv_strides, padding='SAME')

    conv_layer = tf.nn.bias_add(conv_layer, bias)

    conv_layer = tf.nn.relu(conv_layer)

    

    conv_layer = tf.nn.max_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding='SAME')

    

    return conv_layer
def flatten(x_tensor):

    dimension = x_tensor.get_shape().as_list()

    return tf.reshape(x_tensor, [-1, np.prod(dimension[1:])])
def fully_conn(x_tensor, num_outputs):

    dimension = x_tensor.get_shape().as_list()

    shape = list((dimension[-1],) + (num_outputs,))

    weights = tf.Variable(tf.truncated_normal(shape, 0, 0.1))

    bias = tf.Variable(tf.zeros(num_outputs))

    

    fully_conn = tf.nn.relu(tf.add(tf.matmul(x_tensor, weights), bias))

    

    return fully_conn
def output(x_tensor, num_outputs):

    dimension = x_tensor.get_shape().as_list()

    shape = list((dimension[-1],) + (num_outputs,))

    weights = tf.Variable(tf.truncated_normal(shape, 0, 0.01))

    bias = tf.Variable(tf.zeros(num_outputs))

    

    output = tf.add(tf.matmul(x_tensor, weights), bias)

    

    return output
def conv_net(x, keep_prob):

        

    model = conv2d_maxpool(x, conv_num_outputs=32)    

    model = tf.nn.dropout(model, keep_prob)

    

    model = conv2d_maxpool(x, conv_num_outputs=64)    

    model = tf.nn.dropout(model, keep_prob)

    

    model = flatten(model)

    model = tf.nn.dropout(model, keep_prob)

    

    model = fully_conn(model, 128)

    

    model = output(model, 2)

    

    return model



##############################

## Build the Neural Network ##

##############################



# Remove previous weights, bias, inputs, etc..

tf.reset_default_graph()



# Inputs

x = neural_net_image_input((IMG_SIZE, IMG_SIZE, 3))

y = neural_net_label_input()

keep_prob = neural_net_keep_prob()



# Model

logits = conv_net(x, keep_prob)



# Name logits Tensor, so that is can be loaded from disk after training

logits = tf.identity(logits, name='logits')



# Loss and Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)



# Accuracy

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):

    session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})
def print_stats(session, feature_batch, label_batch, cost, accuracy):

    loss = session.run(cost, feed_dict={x:feature_batch, y:label_batch, keep_prob:0.7})

    valid_acc = sess.run(accuracy, feed_dict={

                x: training_img[:batch_size],

                y: training_label[:batch_size],

                keep_prob: 0.7})

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
# TODO: Tune Parameters

epochs = 10

batch_size = 64

keep_probability = 0.7
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

            batch_features = training_img[batch_i:batch_size]

            batch_labels = training_label[batch_i:batch_size]

            

            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)

            

            print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i), end='')

            print_stats(sess, batch_features, batch_labels, cost, accuracy)



    # Save Model

    saver = tf.train.Saver()

    save_path = saver.save(sess, save_model_path)
# Set batch size if not already set

try:

    if batch_size:

        pass

except NameError:

    batch_size = 64



save_model_path = './image_classification'

n_samples = 4

top_n_predictions = 3



def test_model():

    """

    Test the saved model against the test dataset

    """



#     test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))

    loaded_graph = tf.Graph()



    with tf.Session(graph=loaded_graph) as sess:

        # Load model

        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        loader.restore(sess, save_model_path)



        # Get Tensors from loaded model

        loaded_x = loaded_graph.get_tensor_by_name('x:0')

        loaded_y = loaded_graph.get_tensor_by_name('y:0')

        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        

        test_epoch = 25

        # Get accuracy in batches for memory limitations

        test_batch_acc_total = 0

        test_batch_count = 0

        

        n_batches = 500

        b_size = 0

        for i in range(25):

            test_batch_acc_total += sess.run(

                loaded_acc,

                feed_dict={loaded_x: testing_img[b_size:n_batches], loaded_y: testing_label[b_size:n_batches], loaded_keep_prob: 1.0})

            test_batch_count += 1        



            print('Batch {:>2}:  Testing Accuracy: {}\n'.format(i + 1, test_batch_acc_total/test_batch_count), end='')

            

            b_size = n_batches + 1

            n_batches += 500





test_model()