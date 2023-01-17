# Load needed packages

import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import math

from tensorflow.python.framework import ops



# Make plot show inline

%matplotlib inline



# Enable debug feature or not

DEBUG = False
def load_csv2np(train_name, test_name):

    # Load data

    # Convert pandas dataframe to np

    train_raw = pd.read_csv(train_name)

    test_raw = pd.read_csv(test_name)

    train_raw = train_raw.values

    test_raw = test_raw.values

    return train_raw, test_raw



train_raw, test_raw = load_csv2np("../input/train.csv", "../input/test.csv")
# plot image for check purpose

def plot_check(index, data, label):

    print(label[index,:])

    plt.figure()

    plt.imshow(np.reshape(data[index,:], (28,28)))   



# Get train label and normalize data

label = train_raw[:,0:1]      # "0:1" uses to preserve rank

data = train_raw[:,1:]/255.0

test_data = test_raw/255.0



if DEBUG:

    print(label.dtype)

    print(data.dtype)

    print(test_data.dtype)



# Make sure all data are 32-bit wide only

label = label.astype(np.int32)

data = data.astype(np.float32)

test_data = test_data.astype(np.float32)



if DEBUG:

    print(label.dtype)

    print(data.dtype)

    print(test_data.dtype)

    

    # Random 3 pictures for checking    

    for i in np.random.randint(0, label.shape[0], 3):

        plot_check(i, data, label)
# Shuffle train data and split them into train and dev

def shuffle_split(train_org, label_org, dev_pct, seed = 1):

    np.random.seed(seed)

    size = train_org.shape[0]

    dev_size = int(dev_pct * size)

    random_list = np.random.permutation(size)

    train_tmp = train_org[random_list, :]

    label_tmp = label_org[random_list, :]

    train_data = train_tmp[0:dev_size, :]

    train_label = label_tmp[0:dev_size, :]

    dev_data = train_tmp[dev_size:, :]

    dev_label = label_tmp[dev_size:, :]

    return train_data, train_label, dev_data, dev_label



train_data, train_label_raw, dev_data, dev_label_raw = shuffle_split(data, label, 0.2, 10)
print("original data size:", data.shape)

print("train_data size:", train_data.shape)

print("dev_data size:", dev_data.shape)

print("train_label size:", train_label_raw.shape)

print("dev_label size:", dev_label_raw.shape)



if DEBUG:

    # Random 3 pictures for checking    

    for i in np.random.randint(0, train_label_raw.shape[0], 3):

        plot_check(i, train_data, train_label_raw) 



    # Random 3 pictures for checking    

    for i in np.random.randint(0, dev_label_raw.shape[0], 3):

        plot_check(i, dev_data, dev_label_raw) 
# Convert labels to one-hot style

def one_hot(label):

    max_num = np.max(label) + 1

    label = label.reshape(-1) #lower the rank by 1 

    return np.eye(max_num, dtype = np.float32)[label]



train_label = one_hot(train_label_raw)

dev_label = one_hot(dev_label_raw)



if DEBUG:

    print(train_label_raw[10])

    print(train_label[10])

    print(dev_label_raw[48])

    print(dev_label[48])

    print(train_label.shape)

    print(dev_label.shape)
# Define batch normalization

def batch_norm(Z, mean_hist, var_hist, gamma, beta, bn_train_flag, bn_decay = 0.999, epslion = 1e-8):

    if bn_train_flag:

        Zmean, Zvar = tf.nn.moments(Z, [0], keep_dims = True)

        update_mean_hist = tf.assign(mean_hist, bn_decay * mean_hist + (1 - bn_decay) * Zmean)

        update_var_hist = tf.assign(var_hist, bn_decay * var_hist + (1 - bn_decay) * Zvar)

        updates = [update_mean_hist, update_var_hist]

    else:

        Zmean = tf.identity(mean_hist)

        Zvar = tf.identity(var_hist)

        updates = []

    with tf.control_dependencies(updates):

        return (((Z - Zmean)/tf.sqrt(Zvar + epslion)) * gamma) + beta
# Define a function to handle hidden layer

def hidden_layer(inputs, output_size, layer_name, keep_prob_ph, bn_train_ph, bn_decay,

                 tanh_flag = False, drop_flag = False, bn_flag = False):

    with tf.variable_scope(layer_name):

        # Linear layer

        W = tf.get_variable("weight", [int(inputs.shape[1]), output_size], initializer = tf.contrib.layers.xavier_initializer())

        b = tf.get_variable("bias", [1, output_size], initializer = tf.zeros_initializer())

        Ztmp = tf.add(tf.matmul(inputs, W), b, "Ztmp")

        

        # Batch Norm or Not

        if bn_flag:

            gamma = tf.get_variable("gamma", [1, output_size], initializer = tf.ones_initializer())

            beta = tf.get_variable("beta", [1, output_size], initializer = tf.zeros_initializer())

            mean_hist = tf.get_variable("mean_hist", [1, output_size], initializer = tf.zeros_initializer(), 

                                        trainable = False)  

            var_hist = tf.get_variable("var_hist", [1, output_size], initializer = tf.zeros_initializer(), 

                                       trainable = False)

            

            Z = tf.cond(bn_train_ph,

                        lambda: batch_norm(Ztmp, mean_hist, var_hist, gamma, beta, True, bn_decay),

                        lambda: batch_norm(Ztmp, mean_hist, var_hist, gamma, beta, False, bn_decay),

                        name = "Z")

        else:

            Z = tf.identity(Ztmp, "Z")

        

        # Relu layer or tanh layer

        if tanh_flag:

            Atmp = tf.nn.tanh(Z, "Atmp")

        else:

            Atmp = tf.nn.relu(Z, "Atmp")



        # Dropout or Not

        if drop_flag:

            A = tf.nn.dropout(Atmp, keep_prob_ph, name = "A")

        else:

            A = tf.identity(Atmp, "A")



        return A
# Define a function to handle output layer

def output_layer(inputs, output_size, layer_name, bn_train_ph, bn_decay, bn_flag = False):

    with tf.variable_scope(layer_name):

        # Linear layer

        W = tf.get_variable("weight", [int(inputs.shape[1]), output_size], initializer = tf.contrib.layers.xavier_initializer())

        b = tf.get_variable("bias", [1, output_size], initializer = tf.zeros_initializer())

        Ztmp = tf.add(tf.matmul(inputs, W), b, "Ztmp")

        

        # Batch Norm or Not

        if bn_flag:

            gamma = tf.get_variable("gamma", [1, output_size], initializer = tf.ones_initializer())

            beta = tf.get_variable("beta", [1, output_size], initializer = tf.zeros_initializer())

            mean_hist = tf.get_variable("mean_hist", [1, output_size], initializer = tf.zeros_initializer(), 

                                        trainable = False)  

            var_hist = tf.get_variable("var_hist", [1, output_size], initializer = tf.zeros_initializer(), 

                                       trainable = False)

            

            Z = tf.cond(bn_train_ph,

                        lambda: batch_norm(Ztmp, mean_hist, var_hist, gamma, beta, True, bn_decay),

                        lambda: batch_norm(Ztmp, mean_hist, var_hist, gamma, beta, False, bn_decay),

                        name = "Z")

        else:

            Z = tf.identity(Ztmp, "Z")



        return Z
# Define a function to build forward graph

def forward_graph(X_input, hidden_info, tanh_flags_info, bn_flags_info, drop_flags_info, 

                  keep_prob_ph, bn_train_ph, bn_decay):

    no_layer = len(hidden_info)

    A = {}

    A["0"] = X_input



    for i in np.arange(no_layer - 1):

        A[str(i+1)] = hidden_layer(A[str(i)], hidden_info[i], "lay" + str(i+1), keep_prob_ph, bn_train_ph, 

                                   bn_decay, tanh_flags_info[i], drop_flags_info[i], bn_flags_info[i])

    

    return output_layer(A[str(no_layer-1)], hidden_info[no_layer-1], "lay" + str(no_layer), 

                     bn_train_ph, bn_decay, bn_flags_info[no_layer-1])
# Define a function to build cost

def compute_cost(Yhat, Y, lambd_ph, l2_flags_info):

    l2_reg = 0

    

    for index, flag in enumerate(l2_flags_info, start = 1):

        if flag:

            with tf.variable_scope("lay" + str(index), reuse = True):

                if DEBUG:

                    print(tf.get_variable("weight"))

                l2_reg += tf.nn.l2_loss(tf.get_variable("weight"))

    

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Yhat, labels = Y) \

                          + (lambd_ph * l2_reg), name = "cost")
# Define a function to build accuracy

def acc(Yhat, Y):

    prediction = tf.equal(tf.argmax(Yhat, axis = 1), tf.argmax(Y, axis = 1))

    return tf.reduce_mean(tf.cast(prediction, "float"), name = "accuracy")
# Define a function to generate a list of randomize mini batch

def rand_minibatch(data, label, size_minibatch, seed = 1):

    batches_data = []

    batches_label = []

    size = data.shape[0]

    random_list = np.random.permutation(size)

    data_tmp = data[random_list, :]

    label_tmp = label[random_list, :]

    

    no_minibatch = math.floor(size/size_minibatch)

    

    for i in range(0, no_minibatch):

        batch_data = data_tmp[i*size_minibatch:(i+1)*size_minibatch, :]

        batch_label = label_tmp[i*size_minibatch:(i+1)*size_minibatch, :]

        batches_data.append(batch_data)

        batches_label.append(batch_label)

    

    if size/size_minibatch != 0:

        batch_data = data_tmp[no_minibatch*size_minibatch:, :]

        batch_label = label_tmp[no_minibatch*size_minibatch:, :]

        batches_data.append(batch_data)

        batches_label.append(batch_label)

    

    return batches_data, batches_label
# Define a function to train model and predict accuracy

def train(train_data, train_label, dev_data, dev_label, hidden_info, bn_flags_info, l2_flags_info, drop_flags_info, 

          tanh_flags_info, num_epochs = 1000, size_minibatch = 64, learning_rate = 0.0001, lambd_input = 0., 

          keep_prob_input = 1., bn_decay = 0.999, PrintResult = False):

    

    ops.reset_default_graph()       # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)           # to keep consistent results

    data_size = train_data.shape[0]

    no_minibatch = data_size/size_minibatch

    costs_rec = []

    train_acc_rec = []

    dev_acc_rec = []

    

    # Create placefolder for tf

    X = tf.placeholder(tf.float32, [None, train_data.shape[1]], "X")

    Y = tf.placeholder(tf.float32, [None, train_label.shape[1]], "Y")

    lambd = tf.placeholder(tf.float32)  # can be an array so each layer weights have own lambda

    keep_prob = tf.placeholder(tf.float32)  # can be an array so each layer has own keep prob

    bn_train = tf.placeholder(tf.bool)

    

    # Build the forward graph

    Yhat = forward_graph(X, hidden_info, tanh_flags_info, bn_flags_info, drop_flags_info, 

                         keep_prob, bn_train, bn_decay)

    

    # Build cost function

    cost = compute_cost(Yhat, Y, lambd, l2_flags_info)

    

    # Build accuracy estimator

    accuracy = acc(Yhat, Y)

    

    # Backpropagation: Use an AdamOptimizer.

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    # Initialize all the variables

    init = tf.global_variables_initializer()

    

    # Add ops to save and restore all the variables.

    saver = tf.train.Saver()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        rand_seed = 0

        

        # Do the training loop

        for epoch in range(num_epochs):

            

            epoch_cost = 0.                      

            batches_data, batches_label = rand_minibatch(train_data, train_label, size_minibatch, rand_seed)

            rand_seed += 1

            

            for batch_data, batch_label in zip(batches_data, batches_label):

                

                _, batch_cost = sess.run([optimizer, cost], 

                                         feed_dict = {X: batch_data, Y: batch_label, lambd: lambd_input, 

                                                      keep_prob: keep_prob_input, bn_train: True})

                epoch_cost += batch_cost/no_minibatch

            

            # Print the cost every epoch

            if epoch % 5 == 0:

                costs_rec.append(epoch_cost)

            if epoch % 50 == 0:

                train_acc_tmp = accuracy.eval({X: train_data, Y: train_label, lambd: lambd_input, 

                                               keep_prob: 1, bn_train: False})

                dev_acc_tmp = accuracy.eval({X: dev_data, Y: dev_label, lambd: lambd_input, 

                                             keep_prob: 1, bn_train: False})

                train_acc_rec.append(train_acc_tmp)

                dev_acc_rec.append(dev_acc_tmp)

                if PrintResult:

                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))

                    print("Train Accuracy:", train_acc_tmp)

                    print("Dev. Accuracy:", dev_acc_tmp)

                else:

                    print("%i..." %(epoch), end = "")



        # Save vairable for reuse

        print("")

        save_path = saver.save(sess, "./digit_recog.ckpt")

                

    # Plot cost vs epoch

    plt.figure(1)

    plt.plot(np.squeeze(costs_rec))

    plt.ylabel('cost')

    plt.xlabel('per 5 epoch')

    plt.show()

    

    # Plot train_acc, dev_acc per epoch

    plt.figure(2)

    plt.plot(np.squeeze(train_acc_rec)[2:], label = "train")

    plt.plot(np.squeeze(dev_acc_rec)[2:], label = "develop")

    plt.ylabel('Accuracy (%)')

    plt.xlabel('per 50 epoch, start @ 100 epoch')

    plt.legend(loc='lower right')

    plt.show()

    

    return costs_rec, train_acc_rec, dev_acc_rec
def gen_predict(data, hidden_info, bn_flags_info, l2_flags_info, drop_flags_info, tanh_flags_info):

    

    ops.reset_default_graph()       # to be able to rerun the model without overwriting tf variables

    

    # Create placefolder for tf

    X = tf.placeholder(tf.float32, [None, data.shape[1]], "X")

    keep_prob = tf.placeholder(tf.float32)  # can be an array so each layer has own keep prob

    bn_train = tf.placeholder(tf.bool)

    

    # Build the forward graph

    Yhat = forward_graph(X, hidden_info, tanh_flags_info, bn_flags_info, drop_flags_info, 

                         keep_prob, bn_train, bn_decay = 0)



    # Add ops to save and restore all the variables.

    saver = tf.train.Saver()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Load variable in

        saver.restore(sess, "./digit_recog.ckpt")



        # Prediction

        Yhat_output = Yhat.eval({X: data, keep_prob: 1, bn_train: False})

        

        # Prepare matrix for data saving

        class_prediction = Yhat_output.argmax(axis=1)

        class_prediction = np.reshape(class_prediction, [class_prediction.shape[0], 1])



        # Parepare for DataFrame

        pred_data = np.empty([class_prediction.shape[0],2], dtype = np.int32)



        for i, p in enumerate(np.squeeze(class_prediction), start=1):

            pred_data[i-1] = [i, p]

            

        df = pd.DataFrame(pred_data, columns = ["ImageId", "Label"])

        df.to_csv("Digit_Recog.csv", index = False)

        print("Prediction is saved to CSV file")

        

        return class_prediction
# Parameter for NN architecture (predefined to use softmax for output)

# unit per each layer

hidden = [512, 128, 128, 32, 10]

bn_flags = [1, 1, 1, 1, 0]

l2_flags = [1, 1, 1, 1, 1]

drop_flags = [0, 0, 0, 0]

tanh_flags = [1, 1, 1, 1] # False --> Relu



_, _, _ = train(train_data, train_label, dev_data, dev_label, hidden, bn_flags, 

                l2_flags, drop_flags, tanh_flags, num_epochs = 1001, size_minibatch = 64, 

                learning_rate = 0.000005, lambd_input = 0.08, keep_prob_input = 1.0, 

                bn_decay = 0.999, PrintResult = True)
class_prediction = gen_predict(test_data, hidden, bn_flags, l2_flags, drop_flags, tanh_flags)
# Random 5 pictures for checking    

for i in np.random.randint(0, class_prediction.shape[0], 5):

    plot_check(i, test_data, class_prediction)