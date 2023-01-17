import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# First let us load the datasets into different Dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_submission_df = pd.read_csv('../input/sample_submission.csv')

# Dimensions
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
print('Sample submission shape:', sample_submission_df.shape)
# Set of features we have are: date, store, and item
display(train_df.sample(10))
display(test_df.sample(10))
display(sample_submission_df.sample(10))
# Process the training data
vec_train_df = train_df.copy(deep=True)
vec_train_df.date = pd.to_datetime(vec_train_df.date)
vec_train_df.set_index('date', inplace=True)
display(vec_train_df.head())
# Process the test data
vec_test_df = test_df.copy(deep=True)
vec_test_df.date = pd.to_datetime(vec_test_df.date)
vec_test_df.set_index('date', inplace=True)
display(vec_test_df.head())
def vectorized_train_and_test(store, item):
    """
        Returns the final train and test dataframes for the given store and item, ready for training and predicting.
    """
    fltr_train_df = vec_train_df.loc[(vec_train_df.store==store) & (vec_train_df.item==item)]
    fltr_test_df = vec_test_df.loc[(vec_test_df.store==store) & (vec_test_df.item==item)]
    for d in [fltr_train_df, fltr_test_df]:
        d.drop('store', inplace=True, axis=1)
        d.drop('item', inplace=True, axis=1)
    return fltr_train_df,fltr_test_df
tr,_ = vectorized_train_and_test(1,1)
display(tr.head(10))
def generate_train_samples(target_df, batch_size = 10, input_seq_len = 15, output_seq_len = 20):
    x_dates = target_df.index
    total_start_points = len(x_dates) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)
     
    input_seq_x = [list(range(i,(i+input_seq_len))) for i in start_x_idx]
    output_seq_x = [list(range((i+input_seq_len),(i+input_seq_len+output_seq_len))) for i in start_x_idx]
    #print('X Dates')
    #print(start_x_idx)
    #print(input_seq_x)
    #print(output_seq_x)
    
    input_seq_y = [target_df.iloc[i].values for i in input_seq_x]
    output_seq_y = [target_df.iloc[i].values for i in output_seq_x]
    
    ## return shape: (batch_size, time_steps, feature_dims), input_seq_x
    return np.array(input_seq_y), np.array(output_seq_y), input_seq_x
tr_df,_ = vectorized_train_and_test(1,1)
dates = ['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04', '2013-01-05', '2013-01-06', '2013-01-07', '2013-01-08', '2013-01-09']
in_seq_y,out_seq_y,_ = generate_train_samples(tr_df, batch_size = 2, input_seq_len = 2, output_seq_len = 3)
print('Y Sales')
print('In:', np.shape(in_seq_y), ':', in_seq_y)
print('Out:', np.shape(out_seq_y), ':', out_seq_y)
### Below are related RNN
## RNN specific imports
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import copy
## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003 
 
## Network Parameters
# length of input signals
input_seq_len = 60
# length of output signals
output_seq_len = 20
# size of LSTM Cell
hidden_dim = 128
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 10
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5
 
def build_graph(feed_previous = False):
    tf.reset_default_graph()
    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    # Weights and biases
    weights = {
        'out': tf.get_variable('Weights_out',
                               shape = [hidden_dim, output_dim],
                               dtype = tf.float32,
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out',
                               shape = [output_dim],
                               dtype = tf.float32,
                               initializer = tf.constant_initializer(0.)),
    }
    
    # Placeholders
    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]
        
        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]
        
        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]
        
        # The layered network
        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            ##----------------------Helper methods----------------
            def _rnn_decoder(decoder_inputs,
                         initial_state,
                         cell,
                         loop_function=None,
                         scope=None):
                """RNN decoder for the sequence-to-sequence model.
                Args:
                decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                initial_state: 2D Tensor with shape [batch_size x cell.state_size].
                cell: rnn_cell.RNNCell defining the cell function and size.
                loop_function: If not None, this function will be applied to the i-th output
                  in order to generate the i+1-st input, and decoder_inputs will be ignored,
                  except for the first element ("GO" symbol). This can be used for decoding,
                  but also for training to emulate http://arxiv.org/abs/1506.03099.
                  Signature -- loop_function(prev, i) = next
                    * prev is a 2D Tensor of shape [batch_size x output_size],
                    * i is an integer, the step number (when advanced control is needed),
                    * next is a 2D Tensor of shape [batch_size x input_size].
                scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
              Returns:
                A tuple of the form (outputs, state), where:
                  outputs: A list of the same length as decoder_inputs of 2D Tensors with
                    shape [batch_size x output_size] containing generated outputs.
                  state: The state of each cell at the final time-step.
                    It is a 2D Tensor of shape [batch_size x cell.state_size].
                    (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                    states can be the same. They are different for LSTM cells though.) """
                with variable_scope.variable_scope(scope or "rnn_decoder"):
                    state = initial_state
                    outputs = []
                    prev = None
                    for i, inp in enumerate(decoder_inputs):
                        if loop_function is not None and prev is not None:
                            with variable_scope.variable_scope("loop_function", reuse=True):
                                inp = loop_function(prev, i)
                        if i > 0:
                            variable_scope.get_variable_scope().reuse_variables()
                        output, state = cell(inp, state)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = output
                return outputs, state

            #--------------------------------------
            def _basic_rnn_seq2seq(encoder_inputs,
                                   decoder_inputs,
                                   cell,
                                   feed_previous,
                                   dtype=dtypes.float32,
                                   scope=None):
                """Basic RNN sequence-to-sequence model.
                This model first runs an RNN to encode encoder_inputs into a state vector,
                then runs decoder, initialized with the last encoder state, on decoder_inputs.
                Encoder and decoder use the same RNN cell type, but don't share parameters.
                Args:
                  encoder_inputs: A list of 2D Tensors [batch_size x input_size].
                  decoder_inputs: A list of 2D Tensors [batch_size x input_size].
                  feed_previous: Boolean; if True, only the first of decoder_inputs will be
                      used (the "GO" symbol), all other inputs will be generated by the previous
                      decoder output using _loop_function below. If False, decoder_inputs are used
                      as given (the standard decoder case).
                  dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
                  scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
                Returns:
                  A tuple of the form (outputs, state), where:
                    outputs: A list of the same length as decoder_inputs of 2D Tensors with
                      shape [batch_size x output_size] containing the generated outputs.
                    state: The state of each decoder cell in the final time-step.
                      It is a 2D Tensor of shape [batch_size x cell.state_size]. """

                with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                    enc_cell = copy.deepcopy(cell)
                    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

            #--------------------------------------
            def _loop_function(prev, _):
                """Naive implementation of loop function for _rnn_decoder. Transform prev from
                dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
                used as decoder input of next time step """
                return tf.matmul(prev, weights['out']) + biases['out']
            ##----------------------Helper methods----------------
    
        # The seq-seq graph    
        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous = feed_previous
        )
        # The actual output
        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            learning_rate=learning_rate,
            global_step=global_step,
            optimizer='Adam',
            clip_gradients=GRADIENT_CLIPPING)

    # The model saver and restorer
    saver = tf.train.Saver

    # Return the build graph and required tensors
    return dict(
        enc_inp = enc_inp,
        target_seq = target_seq,
        train_op = optimizer,
        loss=loss,
        saver = saver,
        reshaped_outputs = reshaped_outputs,
        )
## Train the model
total_iteractions = 1000
batch_size = 172
KEEP_RATE = 0.5
train_losses = []
val_losses = []

v_train_df,_ = vectorized_train_and_test(1,1)
# For validation phase below
hold_out_len = batch_size
v_train_hold_out_df = v_train_df.tail(hold_out_len)
v_train_df = v_train_df.head(v_train_df.shape[0] - hold_out_len)

rnn_model = build_graph(feed_previous=False)
 
saver = tf.train.Saver()
 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('Model Details')
    print('Total iterations:', total_iteractions)
    print('Training data shape:', v_train_df.shape)
    print('Batch size:', batch_size)
    print('Input sequence len:', input_seq_len)
    print('Output sequence len:', output_seq_len)
    print('--------------------')
    for i in range(total_iteractions):
        batch_input, batch_output,_ = generate_train_samples(target_df=v_train_df, batch_size=batch_size, input_seq_len=input_seq_len, output_seq_len=output_seq_len)
        
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print('Iteration:', i, ' Loss:', loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('seq2seq', 'univariate_ts_model0'))
 
print("Checkpoint saved at: ", save_path)
# Inference/Forecasting
display(v_train_hold_out_df)
test_seq_input,test_seq_output,input_seq_x = generate_train_samples(target_df=v_train_hold_out_df, batch_size=1)
test_seq_input = test_seq_input[0]
print('Test dataframe:')
display(v_train_hold_out_df.iloc[input_seq_x[0]])
print('Expected data:')
last_pos = input_seq_x[0][input_seq_len-1]
print('Last Value:' + str(last_pos))
display(v_train_hold_out_df.iloc[range(last_pos, last_pos + output_seq_len)])
print('test_seq_output', test_seq_output[0])
rnn_model = build_graph(feed_previous=True)
 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = rnn_model['saver']().restore(sess, os.path.join('seq2seq', 'univariate_ts_model0'))
    
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    
    final_preds = np.concatenate(final_preds, axis = 1)
    print('Final predictions:')
    print(final_preds)
