# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline

import math
import time

import tensorflow as tf
# I impletemented with Keras and Tensorflow, in order to learn their difference
# True - run in Keras, False - run in Tensorflow
runInKeras = True
CLASSES = ["cat", "dog"]

IMG_SHAPE = (80, 80, 3)   # (height, width, channels)
from keras.preprocessing import image

def load_image(img_dir, img_name, img_size):
    """
    img_size: (height, width)
    """
    
    img_path = img_dir + img_name
    x = image.load_img(img_path, target_size = img_size)
    x = image.img_to_array(x)
    x = x / 255
#     x = np.expand_dims(x, axis=0)   # For channel = 1
    return x
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input

def load_dataset(img_dir, img_names=None, test_size=None, print_progress=False):
    if img_names is None:
        img_names = os.listdir(img_dir)
        
    m = len(img_names)
    X = np.empty((m, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), dtype=np.float32)
    Y = np.zeros((m, 1))
    
    for i, img_name in enumerate(img_names):
        X[i] = load_image(img_dir, img_name, (IMG_SHAPE[0], IMG_SHAPE[1]))
#         X[i] = preprocess_input(load_image(img_dir, img_name, (IMG_SHAPE[0], IMG_SHAPE[1]))   # For ResNet
        
        if 'dog' in img_name:
            Y[i] = 1
        
        if print_progress and i % 100 == 0:
            print('Processed {0} of {1}'.format(i, m), end='\r')
            
    if test_size is None:
        return X, Y
    else:
        return train_test_split(X, Y, test_size=0.1)
def plot_train_history(history):
    # plot the cost and accuracy 
    loss_list = history['loss']
    val_loss_list = history['val_loss']
    accuracy_list = history['acc']
    val_accuracy_list = history['val_acc']
    # epochs = range(len(loss_list))

    # plot the cost
    plt.plot(loss_list, 'b', label='Training cost')
    plt.plot(val_loss_list, 'r', label='Validation cost')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title('Training and validation cost')
    plt.legend()
    
    plt.figure()
    
    # plot the accuracy
    plt.plot(accuracy_list, 'b', label='Training accuracy')
    plt.plot(val_accuracy_list, 'r', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title('Training and validation accuracy')
    plt.legend()
X_train, X_test, Y_train, Y_test = load_dataset('../input/train/', test_size=0.1, print_progress=True)
print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))
def plot_image(image, label, classes=None):
    img_shape = image.shape   # height, width, channels
    
    if classes is None:
        title = label
    else:
        if not np.isscalar(label):
            label = np.argmax(label)
        title = classes[int(label)]
        
    plt.figure(figsize=(6, 4))
    plt.imshow(np.squeeze(image.reshape(img_shape[0], img_shape[1], img_shape[2])), interpolation='nearest')
    plt.title(title)



if runInKeras:
    image_id = 0
    plot_image(X_test[image_id, :], Y_test[image_id], CLASSES)
def load_img_names(img_dir, test_size=0.1):
    img_names = os.listdir(img_dir)
    m = len(img_names)
    spliter = int(m * (1 - test_size))
    
    permutation = np.random.permutation(m)
    shuffled_img = np.array(img_names)[permutation]
    img_train, img_test = shuffled_img[: spliter], shuffled_img[spliter :]
    
    return img_train, img_test



if not runInKeras:
    img_train, img_test = load_img_names('../input/train/')

    X_test, Y_test = load_dataset('../input/train/', img_names = img_test, print_progress=True)
    print("Test shape: {}".format(X_test.shape))
from keras import backend as K
from keras.layers import Input, Add, Multiply, Average, Maximum, Dense, Activation, ZeroPadding2D, BatchNormalization, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform, he_uniform
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, layer_utils, plot_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.applications.resnet50 import ResNet50
def Res_block(X, filters, kernel_size, strides, name="Res"):
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, kernel_size, strides = strides, padding = 'valid', kernel_initializer = glorot_uniform(seed=0), name = "{}_Conv1".format(name))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(F2, kernel_size, strides = strides, padding = 'same', kernel_initializer = glorot_uniform(seed=0), name = "{}_Conv2".format(name))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, kernel_size, strides = strides, padding = 'valid', kernel_initializer = glorot_uniform(seed=0), name = "{}_Conv3".format(name))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    if tf.shape(X_shortcut)[3] != F3:
        X_shortcut = Conv2D(F3, kernel_size, strides = strides, padding = 'valid', kernel_initializer = glorot_uniform(seed=0), name = "{}_ShortCut".format(name))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
def MyModel(input_shape, num_classes=2):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv1')(X)
#     X = BatchNormalization(axis = 3, name = 'BN1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv2')(X)
#     X = BatchNormalization(axis = 3, name = 'BN2')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv3')(X)
#     X = BatchNormalization(axis = 3, name = 'BN3')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding='same', kernel_initializer=glorot_uniform(), name = 'Conv4')(X)
#     X = BatchNormalization(axis = 3, name = 'BN4')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='MP')(X)


#     Res_model = ResNet50(weights='imagenet', include_top=False, input_tensor=X_input)
#     X = Res_model.output
    
    
    # FLATTEN X (means convert it to a vector)
    X = Flatten()(X)
    
    # FULLY CONNECTED
    X = Dense(128, activation='relu', name='FC1')(X)
    
    if num_classes > 2:
        X = Dense(num_classes, activation='softmax', name='FC2')(X)
    else:
        X = Dense(1, activation='sigmoid', name='FC2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='CNN')

    return model
if runInKeras:
    model = MyModel(IMG_SHAPE, len(CLASSES))
    model.summary()
if runInKeras:
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0003, decay=1e-6, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
if runInKeras:
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', verbose=1, save_best_only=True)   # Save the best model
    hist = model.fit(X_train, Y_train, batch_size=128, callbacks=[monitor, checkpoint], epochs=50, shuffle=True, verbose=1, validation_split=0.01)
if runInKeras:
    plot_train_history(hist.history)
if runInKeras:
    score = model.evaluate(X_test, Y_test)

    print ("Test Loss = " + str(score[0]))
    print ("Test Accuracy = " + str(score[1]))
if runInKeras:
    Y_test_pred = model.predict(X_test, verbose=1)
def create_placeholders(img_shape, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    img_shape -- height, width, number of channels of an input image
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    
    n_H0, n_W0, n_C0 = img_shape
    
    with tf.name_scope("Inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
        Y = tf.placeholder(tf.float32, shape=[None, n_y])
    
    return X, Y
cache = {}

def Conv_Layer(input, filters, kernel_size, strides, keep_prob=1, name="Conv2D", only_Conv=False, skip_MaxPool=False):
    channel_in, channel_out = int(input.shape[3]), filters
    filter_H, filter_W = kernel_size
    stride_H, stride_W = strides
    
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([filter_H, filter_W, channel_in, channel_out], stddev=0.1), name = "{}_W".format(name))
        cache["{}_W".format(name)] = W
        tf.summary.histogram("{}_W".format(name), W)

        conv = tf.nn.conv2d(input, W, strides=[1, stride_H, stride_W, 1], padding="SAME")
        if only_Conv:
            return conv
        
#         tf.nn.batch_normalization(conv, )
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="{}_b".format(name))
        tf.summary.histogram("{}_b".format(name), b)
        
        A = tf.nn.relu(tf.add(conv, b))
        cache["{}_A".format(name)] = A
        tf.summary.histogram("{}_A".format(name), A)
        
        if keep_prob == 1:
            D = A
        else:
            D = tf.nn.dropout(A, keep_prob)
        
        if skip_MaxPool:
            return D
        
        MP = tf.nn.max_pool(D, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        
    return MP
def Dense_Layer(input, units, activation=None, name="Dense"):
    channel_in, channel_out = int(input.shape[1]), units

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channel_in, channel_out], stddev=0.1), name = "{}_W".format(name))
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="{}_b".format(name))
    
        Z = tf.matmul(input, W) + b
    
        tf.summary.histogram("{}_W".format(name), W)
        tf.summary.histogram("{}_b".format(name), b)
        tf.summary.histogram("{}_Z".format(name), Z)
        
        if activation is None:
            return Z
        elif activation == "relu":
            A = tf.nn.relu(Z)
        elif activation == "sigmoid":
            A = tf.nn.sigmoid(Z)
        elif activation == "softmax":
            A = tf.nn.softmax(Z)

    return Z, A
def Res_block(X, filters, kernel_size, strides, name="Res"):
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    X = Conv_Layer(X, filters = F1, kernel_size = kernel_size, strides = strides, name = "{}_Conv1".format(name), skip_MaxPool=True)
    X = Conv_Layer(X, filters = F2, kernel_size = kernel_size, strides = strides, name = "{}_Conv2".format(name), skip_MaxPool=True)
    X = Conv_Layer(X, filters = F3, kernel_size = kernel_size, strides = strides, name = "{}_Conv3".format(name), only_Conv=True)
    
    ##### SHORTCUT PATH ####
    if X_shortcut.shape[3] != F3:
        X_shortcut = Conv_Layer(X, filters = F3, kernel_size = kernel_size, strides = (1, 1), name = "{}_ShortCut".format(name))

    # Add shortcut value to main path, and pass it through a RELU activation
    X = tf.nn.relu(tf.add(X, X_shortcut))
    
    return X
def compute_cost(Z, Y, activation="softmax"):
    if activation == "sigmoid":
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))
    elif activation == "softmax":
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
    
    return cost
def compute_accuracy(Y_pred, Y, activation="softmax"):
    if activation == "sigmoid":
#         correct_prediction = tf.equal(Y_pred > 0.5, tf.cast(Y, tf.bool))
#         correct_prediction = tf.equal(tf.to_float(Y_pred > 0.5), Y)
        correct_prediction = tf.equal(tf.round(Y_pred), Y)
    elif activation == "softmax":
        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy
def forward_propagation(X, num_classes=2):
    """
    Implements the forward propagation for the model
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)

    Returns:
    Z -- the output of the last LINEAR unit
    A -- the output of the last ACTIVATION unit
    """
    
    X = Conv_Layer(X, filters = 32, kernel_size = (3, 3), strides = (1, 1), keep_prob = 0.5, name = "Conv1")
    X = Conv_Layer(X, filters = 64, kernel_size = (3, 3), strides = (1, 1), keep_prob = 0.5, name = "Conv2")
    X = Conv_Layer(X, filters = 128, kernel_size = (3, 3), strides = (1, 1), keep_prob = 0.5, name = "Conv3")
    X = Conv_Layer(X, filters = 128, kernel_size = (3, 3), strides = (1, 1), keep_prob = 0.5, name = "Conv4")
    
#     X = Res_block(X, filters = (32, 32, 64), kernel_size = (3, 3), strides = (1, 1), name = "Res1")
#     X = Res_block(X, filters = (32, 32, 64), kernel_size = (3, 3), strides = (1, 1), name = "Res2")
#     X = Res_block(X, filters = (32, 32, 64), kernel_size = (3, 3), strides = (1, 1), name = "Res3")
    
    # FLATTEN
    X = tf.contrib.layers.flatten(X)
    
    # FULLY CONNECTED
    _, X = Dense_Layer(X, 128, "relu", name = "FC1")
    
    if num_classes > 2:
        Z, A = Dense_Layer(X, 1, "softmax", name = "FC2")
    else:
        #     Z = tf.contrib.layers.fully_connected(F2, 1, activation_fn=None)
        Z, A = Dense_Layer(X, 1, "sigmoid", name = "FC2")

    return Z, A
def random_mini_batches(X, Y, mini_batch_size = 64):
# def random_mini_batches(X, mini_batch_size = 64):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[permutation,:,:,:]
#     shuffled_Y = Y[permutation,:]
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    i_start, i_end = 0, mini_batch_size
    while i_start < m:
        mini_batch_X = shuffled_X[i_start : i_end]
        mini_batch_Y = shuffled_Y[i_start : i_end]

        mini_batches.append((mini_batch_X, mini_batch_Y))
#         mini_batches.append(mini_batch_X)

        i_start += mini_batch_size
        i_end += mini_batch_size
    
    return mini_batches
if not runInKeras:
    learning_rate = 0.0003
    num_epochs = 3
    validation_split = 0.01
    minibatch_size = 128

    min_delta = 1e-3
    patience = 20
    min_cost = None


    spliter = int(len(img_train) * (1-validation_split))
    # X_train_, Y_train_, X_val, Y_val =  X_train[: spliter], Y_train[: spliter], X_train[spliter :], Y_train[spliter :]
    img_train_, img_val = img_train[: spliter], img_train[spliter :]
    X_val, Y_val = load_dataset('../input/train/', img_names = img_val)

    # To keep track of the cost and accuracy
    cost_list, acc_list, val_cost_list, val_acc_list = [], [], [], []


    # to be able to rerun the model without overwriting tf variables
    tf.reset_default_graph()

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(IMG_SHAPE, 1)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z, Y_pred = forward_propagation(X)

    # Add cost and accuracy functions to tensorflow graph
    cost = compute_cost(Z, Y, "sigmoid")
    accuracy = compute_accuracy(Y_pred, Y, "sigmoid")

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    sess = tf.Session()

    # Run the initialization
    sess.run(init)

    # Do the training loop
    for epoch in range(1, num_epochs+1):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        train_cost, train_acc, test_cost, test_acc = 0., 0., 0., 0.

    #     minibatches = random_mini_batches(X_train_, Y_train_, minibatch_size)
        minibatches = random_mini_batches(img_train[: spliter], minibatch_size)
        num_minibatches = len(minibatches) # number of minibatches of size minibatch_size in the train set

        for minibatch in minibatches:
            # Select a minibatch
    #         minibatch_X, minibatch_Y = minibatch
            minibatch_X, minibatch_Y = load_dataset('../input/train/', img_names = minibatch)

            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            _ , minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={X: minibatch_X, Y: minibatch_Y})

            train_cost += minibatch_cost / num_minibatches
            train_acc += minibatch_acc / num_minibatches

        val_cost, val_acc = sess.run([cost, accuracy], feed_dict={X: X_val, Y: Y_val})

        cost_list.append(train_cost)
        acc_list.append(train_acc)
        val_cost_list.append(val_cost)
        val_acc_list.append(val_acc)

        # Ending time.
        end_time = time.time()
        # Difference between start and end-times.
        time_dif = int(round(end_time - start_time))

        # Print the cost every epoch
        print("Epoch %i/%i - %is - cost: %f - acc %f - val_cost: %f - val_acc %f" % (epoch, num_epochs, time_dif, train_cost, train_acc, val_cost, val_acc))

        if min_cost is None or min_cost > val_cost:
            min_cost, min_cost_epoch = val_cost, epoch

        if epoch > min_cost_epoch + patience:
            print("--- Early Stop ---")
            break

    print("--- Completed ---")

    history = {"loss": cost_list, "acc": acc_list, "val_loss": val_cost_list, "val_acc": val_acc_list}
if not runInKeras:
    tf.trainable_variables()

    # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(var)
if not runInKeras:
    plot_train_history(history)
if not runInKeras:
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test}, session=sess)
    print("Test Accuracy:", test_accuracy)
if not runInKeras:
    Y_test_pred = Y_pred.eval({X: X_test, Y: Y_test}, session=sess)
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, classification_report

def analyze(Y, Y_pred, classes, activation="softmax"):
    if activation == "sigmoid":
        Y_cls = Y
        Y_pred_cls = (Y_pred > 0.5).astype(float)
    elif activation == "softmax":
        Y_cls = np.argmax(Y, axis=1)
        Y_pred_cls = np.argmax(Y_pred, axis=1)
    
    
    accuracy = accuracy_score(Y_cls, Y_pred_cls)
    print("Accuracy score: {}\n".format(accuracy))
    
    
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    print("RMSE score: {}\n".format(rmse))

    
    # plot Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(Y_cls, Y_pred_cls)
    print(cm)
    # Plot the confusion matrix as an image.
    plt.matshow(cm)
    # Make various adjustments to the plot.
    num_classes = len(classes)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    
    # plot Classification Report
    print("Classification Report:")
    print(classification_report(Y_cls, Y_pred_cls, target_names=classes))



analyze(Y_test, Y_test_pred, CLASSES, "sigmoid")
def plot_mislabeled(X, Y, Y_pred, classes, activation="softmax", num_images = 0):
    """
    Plots images where predictions and truth were different.
    
    X -- original image data - shape(m, img_rows*img_cols)
    Y -- true labels - eg. [2,3,4,3,1,1]
    Y_pred -- predictions - eg. [2,3,4,3,1,2]
    """
    
    num_col = 5
    
    if activation == "sigmoid":
        Y_cls = Y
        Y_pred_cls = (Y_pred > 0.5).astype(float)
    elif activation == "softmax":
        Y_cls = np.argmax(Y, axis=1)
        Y_pred_cls = np.argmax(Y_pred, axis=1)
    
    mislabeled_indices = np.where(Y_cls != Y_pred_cls)[0]
    
    if num_images < 1:
        num_images = len(mislabeled_indices)
    
    fig, axes = plt.subplots(math.ceil(num_images/num_col), num_col, figsize=(25,20))

    for i, index in enumerate(mislabeled_indices[:num_images]):
#         plt.subplot(2, num_images, i + 1)
#         plt.imshow(X[index, :].reshape(IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), interpolation='nearest')
#         plt.axis('off')
#         plt.title("Prediction: " + classes[p[index]] + " \n Class: " + classes[int(y[index])])
        row, col = i//num_col, i%num_col
        img = np.squeeze(X[index, :].reshape(IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))

        axes[row, col].imshow(img, interpolation='nearest')
        axes[row, col].axis('off')
        axes[row, col].set_title("Id: {}\nPrediction: {} - {}\nClass: {}".format(index, classes[int(Y_pred_cls[index])], np.amax(Y_pred[index]), classes[int(Y_cls[index])]))



plot_mislabeled(X_test, Y_test, Y_test_pred, CLASSES, "sigmoid", 20)
def plot_conv_weights(W):
    _, _, channel_in, channel_out = W.shape
    if channel_out > 10:
        channel_out = 10
    
    fig, axes = plt.subplots(channel_out, channel_in, figsize=(20,20))

    for row in range(channel_out):
        for col in range(channel_in):
            img = W[:, :, col, row]
            axes[row, col].matshow(img, vmin=np.min(W), vmax=np.max(W), interpolation='nearest', cmap='seismic')



            
layer_id = 2

layer_input = model.layers[layer_id]
Conv_W = layer_input.get_weights()[0]

# Retrieve the values of the weight-variables from TensorFlow.
# A feed-dict is not necessary because nothing is calculated.
# Conv_W = sess.run(cache["Conv{}_W".format(layer_id)])
# with tf.variable_scope("Conv1", reuse=True):
#     Conv_W = tf.get_variable('Conv1_W')

print(Conv_W.shape)
plot_conv_weights(Conv_W)
def plot_conv_layer(A):  
    num_col = 5
    
    # Number of filters (channel_out) used in the conv. layer.
    num_filters = A.shape[3]
    if num_filters > 20:
        num_filters = 20
    
    fig, axes = plt.subplots(math.ceil(num_filters/num_col), num_col, figsize=(20,20))

    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            img = A[0, :, :, i]
            ax.matshow(img, interpolation='nearest', cmap='binary')
#             ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])


image_id = 101
layer_id = 3

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_test[image_id : image_id+1])
Conv_A = activations[layer_id]

# Calculate and retrieve the output values of the layer
# when inputting that image.
# Conv_A = sess.run(cache["Conv{}_A".format(layer_id)], {X: X_test[image_id : image_id+1]})

print(Conv_A.shape)
plot_image(X_test[image_id], Y_test[image_id], CLASSES)
plot_conv_layer(Conv_A)
def plot_conv_layers(image, model):
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))

    images_per_row = 16
    
    for layer_name, layer_activation in zip(layer_names, activations):
        if layer_name.startswith('Conv'):
            _, height, width, num_filters = layer_activation.shape   # image height and width, and size of channel
            n_rows = num_filters // images_per_row
            display_grid = np.zeros((n_rows * height, images_per_row * width))

            for row in range(n_rows):
                for col in range(images_per_row):
                    channel_image = layer_activation[0, :, :, row * images_per_row + col]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                    display_grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = channel_image

            plt.figure(figsize=(images_per_row *2, n_rows *2))
            plt.title(layer_name)
            plt.grid(False)
            plt.axis('off')
            plt.imshow(display_grid, aspect='auto', interpolation='nearest', cmap='binary')
            

            
image_id = 101
plot_image(X_test[image_id], Y_test[image_id], CLASSES)
plot_conv_layers(X_test[image_id], model)