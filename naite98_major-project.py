!pip install --upgrade tensorflow==1.14.0

import tensorflow as tf

tf.__version__
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import f1_score, confusion_matrix

from tqdm import tqdm

#import tensorflow as tf

from tensorflow.python.layers.core import Dense

import argparse

import pickle

import sys

import matplotlib.pyplot as plt

import seaborn as sn







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class LSTM_Model():

    def __init__(self, input_shape, lr, a_dim, v_dim, t_dim, emotions, attn_fusion=True, unimodal=False,

                 enable_attn_2=False, seed=1234):

        if unimodal:

            self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))

        else:

            self.a_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], a_dim))

            self.v_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], v_dim))

            self.t_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], t_dim))

        self.emotions = emotions

        self.mask = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0]))

        self.seq_len = tf.placeholder(tf.int32, [None, ], name="seq_len")

        self.y = tf.placeholder(tf.int32, [None, input_shape[0], self.emotions], name="y")

        self.lr = lr

        self.seed = seed

        self.attn_fusion = attn_fusion

        self.unimodal = unimodal

        self.lstm_dropout = tf.placeholder(tf.float32, name="lstm_dropout")

        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.lstm_inp_dropout = tf.placeholder(tf.float32, name="lstm_inp_dropout")

        self.dropout_lstm_out = tf.placeholder(tf.float32, name="dropout_lstm_out")

        self.attn_2 = enable_attn_2



        # Build the model

        self._build_model_op()

        self._initialize_optimizer()



    def GRU(self, inputs, output_size, name, dropout_keep_rate):

        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):

            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)

            bias_init = tf.zeros_initializer()



            cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,

                                          kernel_initializer=kernel_init, bias_initializer=bias_init)

            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_rate)



            output, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)



            return output



    def GRU2(self, inputs, output_size, name, dropout_keep_rate):

        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):

            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)

            bias_init = tf.zeros_initializer()



            fw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,

                                             kernel_initializer=kernel_init, bias_initializer=bias_init)

            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)



            bw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,

                                             kernel_initializer=kernel_init, bias_initializer=bias_init)

            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)



            output_fw, _ = tf.nn.dynamic_rnn(fw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)

            output_bw, _ = tf.nn.dynamic_rnn(bw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)



            output = tf.concat([output_fw, output_bw], axis=-1)

            return output



    def BiGRU(self, inputs, output_size, name, dropout_keep_rate):

        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):

            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)

            bias_init = tf.zeros_initializer()



            fw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,

                                             kernel_initializer=kernel_init, bias_initializer=bias_init)

            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)



            # bw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,

            #                                 kernel_initializer=kernel_init, bias_initializer=bias_init)

            # bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)



            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=fw_cell, inputs=inputs,

                                                         sequence_length=self.seq_len, dtype=tf.float32)



            output_fw, output_bw = outputs

            output = tf.concat([output_fw, output_bw], axis=-1)

            return output



    def self_attention(self, inputs_a, inputs_v, inputs_t, name):

        """



        :param inputs_a: audio input (B, T, dim)

        :param inputs_v: video input (B, T, dim)

        :param inputs_t: text input (B, T, dim)

        :param name: scope name

        :return:

        """



        inputs_a = tf.expand_dims(inputs_a, axis=1)

        inputs_v = tf.expand_dims(inputs_v, axis=1)

        inputs_t = tf.expand_dims(inputs_t, axis=1)

        # inputs = (B, 3, T, dim)

        inputs = tf.concat([inputs_a, inputs_v, inputs_t], axis=1)

        t = inputs.get_shape()[2].value

        share_param = True

        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer

        kernel_init1 = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)

        # kernel_init2 = tf.random_normal_initializer(seed=self.seed, dtype=tf.float32,stddev=0.01)

        # bias_init = tf.zeros_initializer()

        dense = Dense(hidden_size, kernel_initializer=kernel_init1)

        if share_param:

            scope_name = 'self_attn'

        else:

            scope_name = 'self_attn' + name

        # print(scope_name)

        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        with tf.variable_scope(scope_name):

            outputs = []

            for x in range(t):

                t_x = inputs[x, :, :, :]

                # t_x => B, 3, dim

                den = True

                if den:

                    x_proj = dense(t_x)

                    x_proj = tf.nn.tanh(x_proj)

                else:

                    x_proj = t_x

                u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1234))

                x = tf.tensordot(x_proj, u_w, axes=1)

                alphas = tf.nn.softmax(x, axis=-1)

                output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)

                output = tf.squeeze(output, -1)

                outputs.append(output)



            final_output = tf.stack(outputs, axis=1)

            # print('final_output', final_output.get_shape())

            return final_output



    def attention(self, inputs_a, inputs_b, attention_size, params, mask=None, return_alphas=False):

        """

        inputs_a = (b, 18, 100)

        inputs_b = (b, 100)

        :param inputs_a:

        :param inputs_b:

        :param attention_size:

        :param time_major:

        :param return_alphas:

        :return:

        """

        if mask is not None:

            mask = tf.cast(self.mask, tf.bool)

        shared = True

        if shared:

            scope_name = 'attn'

        else:

            scope_name = 'attn_'

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

            hidden_size = inputs_a.shape[2].value  # D value - hidden size of the RNN layer

            den = False

            x_proj = inputs_a

            y_proj = inputs_b

            # print('x_proj', x_proj.get_shape())

            # print('y_proj', y_proj.get_shape())



            # Trainable parameters

            w_omega = params['w_omega']

            b_omega = params['b_omega']

            # dense_attention_2 = params['dense']

            with tf.variable_scope('v', reuse=tf.AUTO_REUSE):

                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;

                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size



                v = tf.tensordot(x_proj, w_omega, axes=1) + b_omega

                # v  = dense_attention_2(x_proj)



            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector

            vu = tf.tanh(tf.matmul(v, tf.expand_dims(y_proj, -1), name='vu'))  # (B,T) shape (B T A) * (B A 1) = (B T)

            vu = tf.squeeze(vu, -1)

            # print('vu', vu.get_shape())

            # masking

            # mask = None

            if mask is not None:

                vu = tf.where(mask, vu, tf.zeros(tf.shape(vu), dtype=tf.float32))



            alphas = tf.nn.softmax(vu, 1, name='alphas')  # (B,T) shape

            if mask is not None:

                alphas = tf.where(mask, alphas, tf.zeros(tf.shape(alphas), dtype=tf.float32))

                a = tf.reduce_sum(tf.expand_dims(alphas, -1), axis=1)

                condition = tf.equal(a, 0.0)

                case_true = tf.ones(tf.shape(a), tf.float32)

                a_m = tf.where(condition, case_true, a)

                alphas = tf.divide(alphas, a_m)



            # print('alphas', alphas.get_shape())



            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

            output = tf.matmul(tf.transpose(inputs_a, [0, 2, 1]), tf.expand_dims(alphas, -1))

            output = tf.squeeze(output, -1)

            # print('r', output.get_shape())

            # output = tf.reduce_sum(r, 1)



            if not return_alphas:

                return tf.expand_dims(output, 1)

            else:

                return tf.expand_dims(output, 1), alphas



    def self_attention_2(self, inputs, name):

        """



        :param inputs_a: audio input (B, T, dim)

        :param inputs_v: video input (B, T, dim)

        :param inputs_t: text input (B, T, dim)

        :param name: scope name

        :return:

        """



        t = inputs.get_shape()[1].value

        share_param = True

        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer

        if share_param:

            scope_name = 'self_attn_2'

        else:

            scope_name = 'self_attn_2' + name

        # print(scope_name)

        # inputs = tf.transpose(inputs, [2, 0, 1, 3])

        # dense = Dense(hidden_size)

        # init1 = tf.random_normal_initializer(seed=self.seed, dtype=tf.float32,stddev=0.01)

        attention_size = hidden_size

        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.01, seed=self.seed))

        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.01, seed=self.seed))

        # dense_attention_2 = Dense(attention_size, activation=None,kernel_initializer=init1,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

        params = {'w_omega': w_omega,

                  'b_omega': b_omega,

                  # 'dense': dense_attention_2

                  }

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

            outputs = []

            for x in range(t):

                t_x = inputs[:, x, :]



                output = self.attention(inputs, t_x, hidden_size, params, self.mask)  # (b, d)

                outputs.append(output)



            final_output = tf.concat(outputs, axis=1)

            return final_output



    def _build_model_op(self):

        # self attention

        if self.unimodal:

            input = self.input

        else:

            if self.attn_fusion:

                input = self.self_attention(self.a_input, self.v_input, self.t_input, '')

                input = input * tf.expand_dims(self.mask, axis=-1)

            else:

                input = tf.concat([self.a_input, self.v_input, self.t_input], axis=-1)



        # input = tf.nn.dropout(input, 1-self.lstm_inp_dropout)

        self.gru_output = self.BiGRU(input, 100, 'gru', 1 - self.lstm_dropout)

        self.inter = tf.nn.dropout(self.gru_output, 1 - self.dropout_lstm_out)

        # self.inter = self.gru_output

        if self.attn_2:

            self.inter = self.self_attention_2(self.inter, '')

        init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)

        if self.unimodal:

            self.inter1 = Dense(100, activation=tf.nn.tanh,

                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(

                self.inter)

        else:

            self.inter1 = Dense(200, activation=tf.nn.relu,

                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(

                self.inter)

            self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)

            self.inter1 = Dense(200, activation=tf.nn.relu,

                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(

                self.inter1)

            self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)

            self.inter1 = Dense(200, activation=tf.nn.relu,

                                kernel_initializer=init, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(

                self.inter1)

        self.inter1 = self.inter1 * tf.expand_dims(self.mask, axis=-1)

        self.inter1 = tf.nn.dropout(self.inter1, 1 - self.dropout)

        self.output = Dense(self.emotions, kernel_initializer=init,

                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))(self.inter1)

        # print('self.output', self.output.get_shape())

        self.preds = tf.nn.softmax(self.output)

        # To calculate the number correct, we want to count padded steps as incorrect

        correct = tf.cast(

            tf.equal(tf.argmax(self.preds, -1, output_type=tf.int32), tf.argmax(self.y, -1, output_type=tf.int32)),

            tf.int32) * tf.cast(self.mask, tf.int32)



        # To calculate accuracy we want to divide by the number of non-padded time-steps,

        # rather than taking the mean

        self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))

        # y = tf.argmax(self.y, -1)



        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.y)

        loss = loss * self.mask



        self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)



    def _initialize_optimizer(self):

        train_vars = tf.trainable_variables()

        reg_loss = []

        total_parameters = 0

        for train_var in train_vars:

            # print(train_var.name)

            reg_loss.append(tf.nn.l2_loss(train_var))



            shape = train_var.get_shape()

            variable_parameters = 1

            for dim in shape:

                variable_parameters *= dim.value

            total_parameters += variable_parameters

        # print(total_parameters)

        print('Trainable parameters:', total_parameters)



        self.loss = self.loss + 0.00001 * tf.reduce_mean(reg_loss)

        self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), dtype=tf.int32,

                                           name='global_step')

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)

        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)



        self.train_op = self._optimizer.minimize(self.loss, global_step=self.global_step)

def multimodal(unimodal_activations, data, classes, attn_fusion=True, enable_attn_2=False):

    print("starting multimodal")

    # Fusion (appending) of features



    text_train = unimodal_activations['text_train']

    audio_train = unimodal_activations['audio_train']

    video_train = unimodal_activations['video_train']



    text_test = unimodal_activations['text_test']

    audio_test = unimodal_activations['audio_test']

    video_test = unimodal_activations['video_test']



    train_mask = unimodal_activations['train_mask']

    test_mask = unimodal_activations['test_mask']



    print('train_mask', train_mask.shape)



    train_label = unimodal_activations['train_label']

    print('train_label', train_label.shape)

    test_label = unimodal_activations['test_label']

    print('test_label', test_label.shape)



    # print(train_mask_bool)

    seqlen_train = np.sum(train_mask, axis=-1)

    print('seqlen_train', seqlen_train.shape)

    seqlen_test = np.sum(test_mask, axis=-1)

    print('seqlen_test', seqlen_test.shape)



    a_dim = audio_train.shape[-1]

    v_dim = video_train.shape[-1]

    t_dim = text_train.shape[-1]

    if attn_fusion:

        print('With attention fusion')

    allow_soft_placement = True

    log_device_placement = False



    # Multimodal model

    session_conf = tf.compat.v1.ConfigProto(

        # device_count={'GPU': gpu_count},

        allow_soft_placement=allow_soft_placement,

        log_device_placement=log_device_placement,

        gpu_options=tf.GPUOptions(allow_growth=True))

    gpu_device = 0

    best_acc = 0

    best_loss_accuracy = 0

    best_loss = 10000000.0

    best_epoch = 0

    best_epoch_loss = 0

    hist = {'epoch': [i for i in range(1,epochs+1)],'trainacc':[],'trainloss':[], 'testacc':[],'testloss':[], 'f1score':[], 'conmat':[]}

    with tf.device('/device:GPU:%d' % gpu_device):

        print('Using GPU - ', '/device:GPU:%d' % gpu_device)

        with tf.Graph().as_default():

            tf.set_random_seed(seed)

            sess = tf.Session(config=session_conf)

            with sess.as_default():

                model = LSTM_Model(text_train.shape[1:], 0.0001, a_dim=a_dim, v_dim=v_dim, t_dim=t_dim,

                                   emotions=classes, attn_fusion=attn_fusion,

                                   unimodal=False, enable_attn_2=enable_attn_2,

                                   seed=seed)

                sess.run(tf.group(tf.global_variables_initializer(),

                                  tf.local_variables_initializer()))



                test_feed_dict = {

                    model.t_input: text_test,

                    model.a_input: audio_test,

                    model.v_input: video_test,

                    model.y: test_label,

                    model.seq_len: seqlen_test,

                    model.mask: test_mask,

                    model.lstm_dropout: 0.0,

                    model.lstm_inp_dropout: 0.0,

                    model.dropout: 0.0,

                    model.dropout_lstm_out: 0.0

                }



                # print('\n\nDataset: %s' % (data))

                print("\nEvaluation before training:")

                # Evaluation after epoch

                step, loss, accuracy = sess.run(

                    [model.global_step, model.loss, model.accuracy],

                    test_feed_dict)

                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(

                    0, step, loss, accuracy))



                for epoch in range(epochs):

                    epoch += 1



                    batches = batch_iter(list(

                        zip(text_train, audio_train, video_train, train_mask, seqlen_train, train_label)),

                        batch_size)



                    # Training loop. For each batch...

                    print('\nTraining epoch {}'.format(epoch))

                    l = []

                    a = []

                    for i, batch in tqdm(enumerate(batches)):

                        b_text_train, b_audio_train, b_video_train, b_train_mask, b_seqlen_train, b_train_label = zip(

                            *batch)

                        # print('batch_hist_v', len(batch_utt_v))

                        feed_dict = {

                            model.t_input: b_text_train,

                            model.a_input: b_audio_train,

                            model.v_input: b_video_train,

                            model.y: b_train_label,

                            model.seq_len: b_seqlen_train,

                            model.mask: b_train_mask,

                            model.lstm_dropout: 0.4,

                            model.lstm_inp_dropout: 0.0,

                            model.dropout: 0.2,

                            model.dropout_lstm_out: 0.2

                        }



                        _, step, loss, accuracy = sess.run([model.train_op, model.global_step,model.loss, model.accuracy],feed_dict)

                        l.append(loss)

                        a.append(accuracy)



                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))

                    hist['trainacc'].append(np.average(a)*100)

                    hist['trainloss'].append(np.average(l)*100)

                    

                    # Evaluation after epoch on test set

                    step, loss, accuracy, preds, y, mask = sess.run([model.global_step, model.loss, model.accuracy,model.preds, model.y, model.mask],test_feed_dict)

                    f1 = f1_score(np.ndarray.flatten(tf.argmax(y, -1, output_type=tf.int32).eval()),

                                  np.ndarray.flatten(tf.argmax(preds, -1, output_type=tf.int32).eval()),

                                  sample_weight=np.ndarray.flatten(tf.cast(mask, tf.int32).eval()), average="weighted")

                    hist['conmat'].append(confusion_matrix(np.ndarray.flatten(tf.argmax(y, -1, output_type=tf.int32).eval()), np.ndarray.flatten(tf.argmax(preds, -1, output_type=tf.int32).eval()), sample_weight=np.ndarray.flatten(tf.cast(mask, tf.int32).eval())))

                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(epoch, step,loss / test_label.shape[0], accuracy, f1))

                    hist['testacc'].append(accuracy*100)

                    hist['testloss'].append((step,loss / test_label.shape[0]) *100)

                    hist['f1score'].append(f1)

                    

                    if accuracy > best_acc:

                        best_epoch = epoch

                        best_acc = accuracy

                    if loss < best_loss:

                        best_loss = loss

                        best_loss_accuracy = accuracy

                        best_epoch_loss = epoch



                print(

                    "\n\nBest epoch: {}\nBest test accuracy: {}\nBest epoch loss: {}\nBest test accuracy when loss is least: {}".format(

                        best_epoch, best_acc, best_epoch_loss, best_loss_accuracy))

    return hist, best_epoch

                

                

#                 print('Sample video testing')

#                 activation_no = 10

#                 test_feed_dict_sample = {

#                     model.t_input: text_test[activation_no:activation_no+1],

#                     model.a_input: audio_test[activation_no:activation_no+1],

#                     model.v_input: video_test[activation_no:activation_no+1],

#                     model.y: test_label[activation_no:activation_no+1],

#                     model.seq_len: seqlen_test[activation_no:activation_no+1],

#                     model.mask: test_mask[activation_no:activation_no+1],

#                     model.lstm_dropout: 0.0,

#                     model.lstm_inp_dropout: 0.0,

#                     model.dropout: 0.0,

#                     model.dropout_lstm_out: 0.0

#                 }

#                 step, loss, accuracy, preds, y, mask = sess.run(

#                     [model.global_step, model.loss, model.accuracy,

#                      model.preds, model.y, model.mask],

#                     test_feed_dict_sample)

#                 #print(y, preds)

#                 print("Video id : ", activation_no)



#                 print("True sentiment : ", tf.argmax(y, axis=-1, output_type=tf.int32).eval(),

#                       "Predicted sentiment : ", tf.argmax(preds, axis=-1, output_type=tf.int32).eval())



#                 print("True sentiment : ", tf.argmax(tf.math.bincount(tf.argmax(y, axis=-1, output_type=tf.int32))).eval(),

#                       "Predicted sentiment : ", tf.argmax(tf.math.bincount(tf.argmax(preds, axis=-1, output_type=tf.int32))).eval())

def batch_iter(data, batch_size, shuffle=True):

    """

    Generates a batch iterator for a dataset.

    """

    data = np.array(data)

    data_size = len(data)

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

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

def plot_train_history(hist):

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('')

    plt.plot(hist['epoch'], hist['acc'], label='Accuracy')

    plt.plot(hist['epoch'], hist['loss'], label = 'Loss')

    plt.legend()
batch_size = 20

epochs = 30

emotions = '6'

data = 'iemocap'

seed = 1234

np.random.seed(seed)

tf.set_random_seed(seed)

unimodal_activations = {}

with open('../input/iemocap/unimodal_{0}_{1}way.pickle'.format(data,emotions),'rb') as handle:

    u = pickle.Unpickler(handle,encoding = 'latin1')

    unimodal_activations = u.load()

hist1, bestEpoch = multimodal(unimodal_activations,data,emotions,False,False)
print("concatenation based fusion")

print("best epoch: ", bestEpoch)

print("Accuracy on test set: ",hist1['testacc'][bestEpoch-1])

print("f1score: ",hist1['f1score'][bestEpoch-1])
df_cm = pd.DataFrame(hist1['conmat'][bestEpoch-1], index = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']],

                  columns = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']])

sn.set(font_scale=1) # for label size

sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap=sn.cm.rocket_r, fmt="d") # font size

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
#test accuracy

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Test Accuracy')

plt.plot(hist1['epoch'], hist1['testacc'], label='Concatenation-based fusion')

plt.legend()



#f1 score

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('f1score')

plt.plot(hist1['epoch'], hist1['f1score'], label='Concatenation-based fusion')

plt.legend()
hist2 , bestEpoch = multimodal(unimodal_activations,data,emotions,True,False)
print("Attention based fusion")

print("best epoch: ", bestEpoch)

print("Accuracy on test set: ",hist2['testacc'][bestEpoch-1])

print("f1score: ",hist2['f1score'][bestEpoch-1])
df_cm = pd.DataFrame(hist2['conmat'][bestEpoch-1], index = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']],

                  columns = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']])

sn.set(font_scale=1) # for label size

sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap=sn.cm.rocket_r, fmt="d") # font size

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
#test accuracy

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Test Accuracy')

plt.plot(hist2['epoch'], hist2['testacc'], label='Attention-based fusion')

plt.legend()



#f1 score

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('f1score')

plt.plot(hist2['epoch'], hist2['f1score'], label='Attention-based fusion')

plt.legend()
hist3 , bestEpoch = multimodal(unimodal_activations,data,emotions,True,True)
print("Attention based fusion with utterance level attention")

print("best epoch: ", bestEpoch)

print("Accuracy on test set: ",hist3['testacc'][bestEpoch-1])

print("f1score: ",hist3['f1score'][bestEpoch-1])
df_cm = pd.DataFrame(hist3['conmat'][bestEpoch-1], index = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']],

                  columns = [i for i in ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']])

sn.set(font_scale=1) # for label size

sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap=sn.cm.rocket_r, fmt="d") # font size

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
#test accuracy

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Test Accuracy')

plt.plot(hist3['epoch'], hist3['testacc'], label='Utterance level attention')

plt.legend()



#f1 score

plt.figure()

plt.xlabel('Epoch')

plt.ylabel('f1score')

plt.plot(hist3['epoch'], hist3['f1score'], label='Utterence level attention')

plt.legend()
plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Test Accuracy')

plt.plot(hist1['epoch'], hist1['testacc'], label='Concatenation-based fusion')

plt.plot(hist2['epoch'], hist2['testacc'], label = 'Attention-based fusion')

plt.plot(hist3['epoch'], hist3['testacc'], label = 'Utterance level attention')

plt.legend()
plt.figure()

plt.xlabel('Epoch')

plt.ylabel('F1score')

plt.plot(hist1['epoch'], hist1['f1score'], label='Concatenation-based fusion')

plt.plot(hist2['epoch'], hist2['f1score'], label = 'Attention-based fusion')

plt.plot(hist3['epoch'], hist3['f1score'], label = 'Utterance level attention')

plt.legend()
plt.figure()

plt.xlabel('Epoch')

plt.ylabel('Train Accuracy')

plt.plot(hist1['epoch'], hist1['trainacc'], label='Concatenation-based fusion')

plt.plot(hist2['epoch'], hist2['trainacc'], label = 'Attention-based fusion')

plt.plot(hist3['epoch'], hist3['trainacc'], label = 'Utterance level attention')

plt.legend()