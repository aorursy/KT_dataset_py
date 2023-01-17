!/opt/conda/bin/python3.7 -m pip install tensorflow==1.15.4





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import random

import tensorflow as tf

from tensorflow.python.keras import backend as K

import os

import sys



print(tf.__version__)
train_mode = True



# if len(sys.argv) > 1:

#    is_test = sys.argv[1]

#    if is_test == "test":

#        train_mode = False



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



#####################################################################################################

#####################################################################################################

    

trainableEmbeddings = True



vocab_size = 30006

embedding_size = 300

hidden_units = 150

batch_size = 16

num_epochs = 3

val_rate = 0.99

max_document_length = 15



train_data_path = "/kaggle/input/k1ueinmslczc2kp/train.csv"

test_data_path = "/kaggle/input/k1ueinmslczc2kp/test.csv" 

save_file = "/kaggle/working/ckp/lstm"

save_path = "/kaggle/working/ckp"



class SiameseLSTMw2v(object):

    

    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):

        n_hidden=hidden_units

        n_layers=1



        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2])) # seq, batch, embed



        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope, reuse=tf.AUTO_REUSE ):

            stacked_rnn_fw = []

            for _ in range(n_layers):

                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)

                stacked_rnn_fw.append(lstm_fw_cell)

            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)



            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)

        return outputs[-1]



    def mean_squared_error(self, y,d,batch_size):

        tmp= tf.square(y - d)

        return tf.reduce_sum(tmp)/(2 * batch_size)

        #return tf.reduce_sum(tmp)/(batch_size)



    def __init__(

        self, sequence_length, vocab_size, embedding_size, hidden_units, batch_size, trainableEmbeddings):



        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")

        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")

        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")



          

        with tf.name_scope("embedding"):

            self.W = tf.Variable(

                tf.random_normal([vocab_size, embedding_size],  stddev=0.1),

                trainable=trainableEmbeddings,name="W")

            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)

            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)





        with tf.name_scope("output"):

            self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "shared", embedding_size, sequence_length, hidden_units)

            self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "shared", embedding_size, sequence_length, hidden_units)



            #Manhattan Distance

            #self.distance = K.exp(-K.sum(K.abs(self.out1 - self.out2), axis=1, keepdims=True))

            self.distance = tf.math.exp(-1*tf.math.reduce_sum(tf.math.abs(self.out1 - self.out2), axis=1, keepdims=True))

            self.distance = tf.reshape(self.distance, [-1], name="distance")



        with tf.name_scope("loss"):

            self.loss = self.mean_squared_error(self.input_y,self.distance, batch_size)



        with tf.name_scope("accuracy"):

            self.temp_sim = tf.rint(self.distance, name="temp_sim") #auto threshold 0.5

            correct_predictions = tf.equal(self.temp_sim, self.input_y)

            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")







#####################################################################################################

#####################################################################################################



def get_train_data_from_csv(path):

    train_data = pd.read_csv(path)



    x1_data = []

    x2_data = []

    y_data = []

    for d in train_data["sentence1"]:

        x1_data.append(d.split())

    

    for d in train_data["sentence2"]:

        x2_data.append(d.split())

    

    for d in train_data["label"]:

        y_data.append(d)



    return x1_data, x2_data, y_data



def get_test_data_from_csv(path):

    test_data = pd.read_csv(path)

    id_col = test_data['id']



    ID = []

    for i in id_col:

        ID.append(i)



    x1_data = []

    x2_data = []

    for d in test_data["sentence1"]:

        x1_data.append(d.split())

    

    for d in test_data["sentence2"]:

        x2_data.append(d.split())



    return x1_data, x2_data, ID 







def transform(arrs, max_document_length):

    transformed = []

    for arr in arrs:

        word_ids = np.zeros(max_document_length, np.int64)

        for idx, val in enumerate(arr):

            if idx >= max_document_length:

                break

            word_ids[idx] = int(val)

        transformed.append(word_ids)

    return np.asarray(transformed)





def batch_iter(data, batch_size, num_epochs, shuffle=True):

    """

    Generates a batch iterator for a dataset.

    """

    data = np.asarray(data)

    #print(data)

    #print(data.shape)

    data_size = len(data)

    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):

        # Shuffle the data at each epoch

        if shuffle:

            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_data = data[shuffle_indices]

        else:

            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):

            start_index = batch_num * batch_size

            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]

 

if train_mode:

     

    x1_data, x2_data, y_data = get_train_data_from_csv(train_data_path)

    

    

    trans_x1 = transform(x1_data, max_document_length)

    trans_x2 = transform(x2_data, max_document_length)

    

    #split train, vaildation

    train_x1 = trans_x1[:int(len(trans_x1)*val_rate)]

    train_x2 = trans_x2[:int(len(trans_x2)*val_rate)]

    train_y = y_data[:int(len(y_data)*val_rate)]

    

    dev_x1 = trans_x1[int(len(trans_x1)*val_rate):]

    dev_x2 = trans_x2[int(len(trans_x2)*val_rate):]

    dev_y = y_data[int(len(y_data)*val_rate):]

    

    

    batches=batch_iter(list(zip(train_x1, train_x2, train_y)), batch_size, num_epochs)

    

    siameseModel = SiameseLSTMw2v(

        sequence_length=max_document_length,

        vocab_size=vocab_size,

        embedding_size=embedding_size,

        hidden_units=hidden_units,

        batch_size=batch_size,

        trainableEmbeddings=trainableEmbeddings

    )

    

    global_step = tf.Variable(0, name="global_step", trainable=False)

    

    iter_ = int(len(train_x1) / batch_size)

    

    #lr = 1e-2

    #learning_rate = tf.train.exponential_decay(lr, global_step,

    #                                           iter_, 0.75, staircase=True)

    #optimizer = tf.train.AdamOptimizer(learning_rate)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    

    optimizer = tf.train.AdamOptimizer(1e-3)

    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)

    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    

    sess = tf.Session()

    

    sess.run(tf.global_variables_initializer())

    

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

    

    checkpoint_every = 100

    max_validation_acc = 0

    

    for ep in range(num_epochs):

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        print("Epoch: %d" % (ep+1))

        for it in range(iter_):

            batch = batches.__next__()

            if len(batch)<1:

                continue



            x1_batch,x2_batch, y_batch = zip(*batch)

            if len(y_batch)<1:

                continue

            

            feed_dict = {

                siameseModel.input_x1: x1_batch,

                siameseModel.input_x2: x2_batch,

                siameseModel.input_y: y_batch,

                siameseModel.dropout_keep_prob: 1.0,

            }

    

            import time

            time_str = time.time()

            _, step, loss, accuracy, dist, sim = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim],  feed_dict)

            time_str = time.time() - time_str

    

    

            if step % checkpoint_every == 0:

                print("###########################")

                print(y_batch)

                print("dist")

                print(dist)

                print("sim")

                print(sim)

    

                feed_dict_dev = {

                    siameseModel.input_x1: dev_x1,

                    siameseModel.input_x2: dev_x2,

                    siameseModel.input_y: dev_y,

                    siameseModel.dropout_keep_prob: 1.0,

                }

                dev_loss, dev_accuracy, dev_dist, dev_sim = sess.run([siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim],  feed_dict_dev)

                print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, dev_loss, dev_accuracy))

                if max_validation_acc < dev_accuracy and dev_accuracy > 0.81:

                   max_validation_acc = dev_accuracy

                   saver.save(sess, save_file, global_step=step)

                   print("Saved model {} with max_accuracy={} checkpoint to {}\n".format(ep, accuracy, "lstm"))



#else: #make test output

    

    x1_data, x2_data, ID = get_test_data_from_csv(test_data_path)



    trans_x1 = transform(x1_data, max_document_length)

    trans_x2 = transform(x2_data, max_document_length)



#     siameseModel = SiameseLSTMw2v(

#         sequence_length=max_document_length,

#         vocab_size=vocab_size,

#         embedding_size=embedding_size,

#         hidden_units=hidden_units,

#         batch_size=batch_size,

#         trainableEmbeddings=trainableEmbeddings

#     )



#     global_step = tf.Variable(0, name="global_step", trainable=False)



#     sess = tf.Session()

    

#     sess.run(tf.global_variables_initializer())



#     saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)



    saver.restore(sess,tf.train.latest_checkpoint(save_path))





    feed_dict = {

        siameseModel.input_x1: trans_x1,

        siameseModel.input_x2: trans_x2,

        siameseModel.dropout_keep_prob: 1.0,

    }



    import time

    time_str = time.time()

    step, dist, sim = sess.run([global_step, siameseModel.distance, siameseModel.temp_sim],  feed_dict)

    

    time_str = time.time() - time_str



    print("TEST {}: step {}".format(time_str, step))

    print("dist")

    print(dist)

    print("sim")

    print(sim)

    

    sim = [int(x) for x in sim]

    

    data = {'id':ID,'label':sim}

    last_output = pd.DataFrame(data)

    

    print(last_output)

    last_output.to_csv("/kaggle/working/gj_result.csv", mode='w', index=False)

    #last_output.to_csv("./gj_result.csv", mode='w', index=False)



    sess.close()

    #input("End: press Enter to finish session")

    




