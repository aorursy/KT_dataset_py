# system imports

import os

import sys

import random

import collections



# data imports

import numpy as np





# tensorflow imports

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras import initializers

from tensorflow.python.keras import regularizers

from tensorflow.python.keras import constraints

from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.keras import backend as K

from tensorflow.python.ops import array_ops

from tensorflow.python.framework import constant_op

from tensorflow.python.ops import gen_cudnn_rnn_ops

from tensorflow.python.framework import ops

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin

from tensorflow.python.util import nest

from tensorflow.python.keras import activations

from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras.utils import generic_utils

from tensorflow.python.keras.utils import tf_utils



from tensorflow.python.distribute import distribution_strategy_context as ds_context

from tensorflow.python.eager import context
class LSTMCell(Layer):

    def __init__(self,units):

        super(LSTMCell, self).__init__()

        self.units = units

        

        # specify some parameters

        self.activation = activations.get('tanh')

        self.recurrent_activation = activations.get('hard_sigmoid')

        self.kernel_initializer = initializers.get('glorot_uniform')

        self.recurrent_initializer = initializers.get('orthogonal')

        self.bias_initializer = initializers.get('zeros')



        # two of the requirements to consider this layer a cell

        self.state_size = data_structures.NoDependency([self.units, self.units])

        self.output_size = self.units





    def build(self, input_shape):

        input_dim = input_shape[-1]

        

        # initialize the weights .. remember I am creating four times of weights

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4), name='kernel', initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), name='recurrent_kernel', initializer=self.recurrent_initializer)





    def call(self, inputs, states, training=None):

        

        # h(t-1) and c(t-1)

        h_tm1 = states[0]  # previous memory state

        c_tm1 = states[1]  # previous carry state



        # copy the inputs four times , go to the equations, you will find xt used four times (f, i, c, o)

        inputs_i = inputs

        inputs_f = inputs

        inputs_c = inputs

        inputs_o = inputs

        

        # split the weights into the four neurons, go to the build function, you find I created them all together, this is time to split

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)

        

        # multiple the weights by each input for each gate (f, i, c, o), in the equations above, you will find xt multiplied by each their corresponding weights(Wf, Wi, Wc, Wo).

        x_i = K.dot(inputs_i, k_i)

        x_f = K.dot(inputs_f, k_f)

        x_c = K.dot(inputs_c, k_c)

        x_o = K.dot(inputs_o, k_o)



        # copy h(t-1) four times, in the above equations, h(t-1) is used four times in (f, i, c, o) equations.

        h_tm1_i = h_tm1

        h_tm1_f = h_tm1

        h_tm1_c = h_tm1

        h_tm1_o = h_tm1

        

        

        # equation i

        i = self.recurrent_activation(x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units])) 

        

        # equation f

        f = self.recurrent_activation(x_f + K.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))

        

        # equation c

        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))

        

        # equation o

        o = self.recurrent_activation(x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))



        



        # calculate h(t) the output, h(t) will also be the memory state, and c is the carry state  (look above you will see three inputs and three outputs)

        h = o * self.activation(c)

        return h, [h, c]



    # one of the requirements to consider this layer a cell

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        return list(_generate_zero_filled_state(batch_size, self.state_size, dtype))



class RNN(Layer):

    

    def __init__(self,cell,return_sequences=False,return_state=False):

        super(RNN, self).__init__()

        self.cell = cell

        self.return_sequences = return_sequences

        self.return_state = return_state

        self.input_spec = None 

        self.state_spec = None



    def build(self, input_shape):

        # Get the shape of the input ,many stuff to take care of any error, in the end- We will create two InputSpecs input_spec and state_spec and it will call the build of the LSTMCell which will prepare

        # it as well by instantiating the weights , check the build() in LSTMCell above.

        if isinstance(input_shape, list):

            input_shape = input_shape[0]



        def get_input_spec(shape):

            input_spec_shape = shape.as_list()

            input_spec_shape[0] = None

            input_spec_shape[1] = None

            return InputSpec(shape=tuple(input_spec_shape))



        def get_step_input_shape(shape):

            shape = tuple(shape.as_list())

            return (shape[0],) + shape[2:]



        input_shape = tensor_shape.as_shape(input_shape)

        if self.input_spec is not None:

            self.input_spec[0] = get_input_spec(input_shape)

        else:

            self.input_spec = [get_input_spec(input_shape)]

            

        step_input_shape = get_step_input_shape(input_shape)

        self.cell.build(step_input_shape)



        if _is_multiple_state(self.cell.state_size):

            state_size = list(self.cell.state_size)

        else:

            state_size = [self.cell.state_size]



        self.state_spec = [InputSpec(shape=[None] + tensor_shape.as_shape(dim).as_list()) for dim in state_size]

        print("input_spec", self.input_spec)

        print("state_spec", self.state_spec)

        self.built = True



    def __call__(self, inputs, initial_state=None):

        inputs, initial_state = _standardize_args(inputs,initial_state)

        if initial_state is None:

            return super(RNN, self).__call__(inputs)

        additional_inputs = []

        additional_specs = []

        if initial_state is not None:

            additional_inputs += initial_state

            self.state_spec = nest.map_structure(

              lambda s: InputSpec(shape=K.int_shape(s)), initial_state)

            additional_specs += self.state_spec



        # additional_inputs can be empty if initial_state is provided

        # but empty (e.g. the cell is stateless).

        flat_additional_inputs = nest.flatten(additional_inputs)

        is_keras_tensor = K.is_keras_tensor(

            flat_additional_inputs[0]) if flat_additional_inputs else True



        if is_keras_tensor:

            # Compute the full input spec, including state and constants

            full_input = [inputs] + additional_inputs

            # The original input_spec is None since there could be a nested tensor

            # input. Update the input_spec to match the inputs.

            full_input_spec = generic_utils.to_list(

              nest.map_structure(lambda _: None, inputs)) + additional_specs

            # Perform the call with temporarily replaced input_spec

            self.input_spec = full_input_spec

            output = super(RNN, self).__call__(full_input)

            # Remove the additional_specs from input spec and keep the rest. It is

            # important to keep since the input spec was populated by build(), and

            # will be reused in the stateful=True.

            self.input_spec = self.input_spec[:-len(additional_specs)]

            return output

        else:

            return super(RNN, self).__call__(inputs)



        

    def call(self,inputs,initial_state=None):

        if (isinstance(inputs, collections.Sequence) and not isinstance(inputs, tuple)):

            initial_state = inputs[1:]

            if len(initial_state) == 0:

                initial_state = None

            inputs = inputs[0]

        

        if initial_state is None:

            input_shape = array_ops.shape(inputs)

            initial_state = self.cell.get_initial_state(inputs=None, batch_size=input_shape[0], dtype=inputs.dtype)

                    

            if not nest.is_sequence(initial_state): # Keras RNN expect the states in a list, even if it's a single state tensor.

                initial_state = [initial_state]       

   

        input_shape = K.int_shape(inputs)

        timesteps = input_shape[1]



        def step(inputs, states):

            output, new_states = self.cell.call(inputs, states)

            if not nest.is_sequence(new_states):

                new_states = [new_states]

            return output, new_states



        # I stopped here, .. so the loop will be done here by K.rnn (https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/backend.py#L3809)

        # to save the states and outputs..........

        last_output, outputs, states = K.rnn(step,inputs,initial_state)



        if self.return_sequences:

            output = outputs

        else:

            output = last_output

            

        if self.return_state:

            return generic_utils.to_list(output) + list(states)

        else:

            return output
# some helper functions

def _standardize_args(inputs, initial_state):

    if isinstance(inputs, list):

        assert initial_state is None

        if len(inputs) > 1:

            initial_state = inputs[1:]

            inputs = inputs[:1]

        if len(inputs) > 1:

            inputs = tuple(inputs)

        else:

            inputs = inputs[0]



    def to_list_or_none(x):

        if x is None or isinstance(x, list):

            return x

        if isinstance(x, tuple):

            return list(x)

        return [x]



    initial_state = to_list_or_none(initial_state)

    return inputs, initial_state





def _is_multiple_state(state_size):

    return (hasattr(state_size, '__len__') and

          not isinstance(state_size, tensor_shape.TensorShape))





def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):

    def create_zeros(unnested_state_size):

        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()

        init_state_size = [batch_size_tensor] + flat_dims

        return array_ops.zeros(init_state_size, dtype=dtype)



    if nest.is_sequence(state_size):

        return nest.map_structure(create_zeros, state_size)

    else:

        return create_zeros(state_size)
input_texts = []

target_texts = []

input_characters = set()

target_characters = set()

num_samples = 10000



with open('../input/french-english-translation-for-seq2seq-model/fra.txt', 'r', encoding='utf-8') as f:

    counter = 0

    for line in f:

        input_text, target_text,_ = line.split('\t')     

        

        target_text = '\t' + target_text + '\n'

        input_texts.append(input_text)

        

        target_texts.append(target_text)

        

        for char in input_text:

            if char not in input_characters:

                input_characters.add(char)

        

        for char in target_text:

            if char not in target_characters:

                target_characters.add(char)

        

        counter = counter + 1

        if counter == num_samples :

            break;

            

#  characters and the number of them

input_characters = sorted(list(input_characters))

target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)

num_decoder_tokens = len(target_characters)



# maximum number of words in a sentence for both input and output

max_encoder_seq_length = max([len(txt) for txt in input_texts])

max_decoder_seq_length = max([len(txt) for txt in target_texts])



# indexing i.e. for each character use a number to refer to it

input_token_index = {char: i for i, char in enumerate(input_characters)}

target_token_index = {char: i for i, char in enumerate(target_characters)}



# encoder_input_data.shape is (175623, 262, 91) i.e create an array of all the text and pad it to the maximum length

# for both the encoder and the decoder, each sentence will be represended by 262 words and each word will be represented 

# with 91 characters. e.g. (cat is good), the first three of 262 will be filled everything else will be zero. For the first

# (cat) only three letters of the 91 will take 1 and everything else is 0.



encoder_input_data = np.zeros(

    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),

    dtype='float32')



# two decoder data input and target



decoder_input_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')

decoder_target_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')









for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for t, char in enumerate(input_text):

        encoder_input_data[i, t, input_token_index[char]] = 1.

    for t, char in enumerate(target_text):

        # decoder_target_data is ahead of decoder_input_data by one timestep

        decoder_input_data[i, t, target_token_index[char]] = 1.

        if t > 0:

            # decoder_target_data will be ahead by one timestep

            # and will not include the start character.

            decoder_target_data[i, t - 1, target_token_index[char]] = 1.



print('Number of samples:', len(input_texts))

print('Number of unique input tokens:', num_encoder_tokens)

print('Number of unique output tokens:', num_decoder_tokens)

print('Max sequence length for inputs:', max_encoder_seq_length)

print('Max sequence length for outputs:', max_decoder_seq_length)

print("encoder_input_data shape:",encoder_input_data.shape)

print("decoder_input_data shape:",decoder_input_data.shape)

print("decoder_target_data shape:",decoder_target_data.shape)



# Reverse-lookup token index to decode sequences back to

# something readable.

reverse_input_char_index = dict(

    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(

    (i, char) for char, i in target_token_index.items())



latent_dim = 256

#from tensorflow.keras.layers import RNN

# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))

print("build() called for the first RNN ")

encoder = RNN(LSTMCell(latent_dim), return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]



decoder_inputs = Input(shape=(None, num_decoder_tokens)) # Set up the decoder, using `encoder_states` as initial state.



print("build() called for the second RNN ")

decoder_lstm = RNN(LSTMCell(latent_dim), return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)



decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)



model = Model([encoder_inputs, decoder_inputs], decoder_outputs) # the last state will be fed to the decoder



optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001)

model.compile(optimizer, loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

          batch_size = 64,

          epochs = 20,

          validation_split=0.2)









encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(

    decoder_inputs, initial_state= decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)





decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)



input_seq = encoder_input_data[2019:2020]

# Encode the input as state vectors.

states_value = encoder_model.predict(input_seq)



# Generate empty target sequence of length 1.

target_seq = np.zeros((1, 1, num_decoder_tokens))

# Populate the first character of target sequence with the start character.

target_seq[0, 0, target_token_index['\t']] = 1.



# Sampling loop for a batch of sequences

# (to simplify, here we assume a batch of size 1).

stop_condition = False

decoded_sentence = ''

while not stop_condition:

    output_tokens, h, c = decoder_model.predict(

        [target_seq] + states_value)



    # Sample a token

    sampled_token_index = np.argmax(output_tokens[0, -1, :])

    sampled_char = reverse_target_char_index[sampled_token_index]

    decoded_sentence += sampled_char



    # Exit condition: either hit max length

    # or find stop character.

    if (sampled_char == '\n' or

       len(decoded_sentence) > max_decoder_seq_length):

        stop_condition = True



    # Update the target sequence (of length 1).

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    target_seq[0, 0, sampled_token_index] = 1.



    # Update states

    states_value = [h, c]



print('-')

print('Input sentence:', input_texts[2019])

print('Decoded sentence:', decoded_sentence)