# Note: this example does not cover tokenization.



# First we build the encoder.

#

# First, generate word embeddings for the input tokens. To learn more about word embeddings read the

# following notebook: https://www.kaggle.com/residentmario/notes-on-word-embedding-algorithms/

#

# In this example training the word embeddings is part of the training process. You can use a prebuilt

# volcabulary instead, if you'd like.

#

# mask_zero is set to True to signal that the value 0 is to be used as a mask value in the dataset.

# In that case note that 0 becomes a reserved value in the token dictionary, so you'll need to add

# a +1 to your token dictionary size.

#

# encoder_input_values  = [['token', 'list', 'to', 'be', 'processed', 'in', 'input', 'language']]

encoder_input = Input(shape=(INPUT_LENGTH,))

encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)



# Now the encoder's LSTM layer. return_sequences=True has the LSTM return all of the hidden vectors

# passed between time-steps, instead of just the last (output) hidden vector. unroll is a speedup

# optimization that is only effective for small input sizes.

encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)



# Now we build the decoder. Again an embedding layer is the first layer.

# 

# Recall that in our model: sentence in one lag -> encoder -> information representation ->

# decoder -> sentence in another lang. So the embedding layer is receiving a discretely vectorized

# representation of the information contained in the input sentence. You can think of this as

# a list of approximately correct word vectors in the target language, that the decoder then has to

# "lint". Cf. https://i.imgur.com/R3HZK2k.png

decoder_input = Input(shape=(OUTPUT_LENGTH,))

decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)



# Now the decoder's LSTM layer. The main different here is that initial_state is initialized with

# initial_state of the weights set to a non-default value: the final weights discovered by

# the encoder model.

encoder_last = encoder[:,-1,:]

decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])



# --------------------------------------------------------------------

# Our target output is a sentence of a certain length. We want the model to vote on the likelihood

# of every individual word in the dictionary as being the next word in the output. E.g. there is one

# probability assigned per token per sentence position. This requires having one fully connected

# (Dense) node per word, e.g. output_dict_size nodes, in a TimeDistributed blanket.



# the Dense layer propogate across the OUTPUT_LENGTH time steps outputs we got from the immediately previous

# LSTM(64, return_sequences=True).

output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)



# Finally we define the model.

#

# Model training solves the encoder and the decoder simultaneously, so we have to set the model

# construction likewise.

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])

model.compile(optimizer='adam', loss='binary_crossentropy')



model.fit(

    x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],

    verbose=2,

    batch_size=64,

    epochs=60

)



# The model has to be applied to data sequentially, with prior tokens fed into the

# decoder module (and the newly contributing neural connections unmasked) as we

# proceed through the task. The code that does this in this particular project,

# taken from the project repo

# (https://github.com/wanasit/katakana/blob/master/katakana/model.py#L92), is as

# follows:

def to_katakana(text, model, input_encoding, output_decoding,

                input_length=DEFAULT_INPUT_LENGTH,

                output_length=DEFAULT_OUTPUT_LENGTH):



    encoder_input = encoding.transform(input_encoding, [text.lower()], input_length)

    decoder_input = np.zeros(shape=(len(encoder_input), output_length))

    decoder_input[:, 0] = encoding.CHAR_CODE_START

    for i in range(1, output_length):

        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)

        decoder_input[:, i] = output[:, i]



    decoder_output = decoder_input

    return encoding.decode(output_decoding, decoder_output[0][1:])
encoder_input = Input(shape=(INPUT_LENGTH,))

encoder = Embedding(input_dict_size, 64, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)

encoder = LSTM(64, return_sequences=True, unroll=True)(encoder)



encoder_last = encoder[:,-1,:]



decoder_input = Input(shape=(OUTPUT_LENGTH,))

decoder = Embedding(output_dict_size, 64, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)

decoder = LSTM(64, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])



# --------------------------------------------------------------------

# The new attention code follow.



from keras.layers import Activation, dot, concatenate

attention = dot([decoder, encoder], axes=[2, 2])

attention = Activation('softmax')(attention)

context = dot([attention, encoder], axes=[2,1])

decoder_combined_context = concatenate([context, decoder])

output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)

output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)



# --------------------------------------------------------------------

# Compare with the old code:

# output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)

# --------------------------------------------------------------------



model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(

    x=[training_encoder_input, training_decoder_input],

    y=[training_decoder_output],

    verbose=2,

    batch_size=64,

    epochs=60

)
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
batch_size = 64

epochs = 100

num_samples = 10000



# "Latent dimensionality" of the encoding space --- aka the length

# of the hidden word vectors.

latent_dim = 256



# This model assumes that our inputs are already tokenized. In fact in the

# original source code the inputs are individual *characters*, not word

# tokens, as would be typical in application. This was done for the sake

# of demonstrative simplicity.

encoder_inputs = Input(shape=(None, num_encoder_tokens))



# return_state=True means that the hidden and cell state vectors of the final

# layer of the LSTM are returned, in addition to the output vector. As in the

# previous model, the weights on the LSTM output layer in the encoder model

# are used to seed the weights in the first layer of the LSTM input layer in the

# decoder model.

encoder = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]



decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(

    decoder_inputs,

    # setting the initial_state value here as before is what creates the

    # encoder-decoder dependency

    initial_state=encoder_states

)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)



# Now define the model and train it.

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(

    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']

)

model.fit(

    [encoder_input_data, decoder_input_data], decoder_target_data,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.2

)

model.save('s2s.h5')



# This model has a similar sequential stepping mechanism to its prediction

# application, omitted for the sake of brevity. Check the source code for that.
from math import log

from numpy import array

from numpy import argmax

 

# beam search

def beam_search_decoder(data, k):

	sequences = [[list(), 1.0]]

	# walk over each step in sequence

	for row in data:

		all_candidates = list()

		# expand each current candidate

		for i in range(len(sequences)):

			seq, score = sequences[i]

			for j in range(len(row)):

				candidate = [seq + [j], score * -log(row[j])]

				all_candidates.append(candidate)

		# order all candidates by score

		ordered = sorted(all_candidates, key=lambda tup:tup[1])

		# select k best

		sequences = ordered[:k]

	return sequences

 

# define a sequence of 10 words over a vocab of 5 words

data = [[0.1, 0.2, 0.3, 0.4, 0.5],

		[0.5, 0.4, 0.3, 0.2, 0.1],

		[0.1, 0.2, 0.3, 0.4, 0.5],

		[0.5, 0.4, 0.3, 0.2, 0.1],

		[0.1, 0.2, 0.3, 0.4, 0.5],

		[0.5, 0.4, 0.3, 0.2, 0.1],

		[0.1, 0.2, 0.3, 0.4, 0.5],

		[0.5, 0.4, 0.3, 0.2, 0.1],

		[0.1, 0.2, 0.3, 0.4, 0.5],

		[0.5, 0.4, 0.3, 0.2, 0.1]]

data = array(data)

# decode sequence

result = beam_search_decoder(data, 3)

# print result

for seq in result:

	print(seq)