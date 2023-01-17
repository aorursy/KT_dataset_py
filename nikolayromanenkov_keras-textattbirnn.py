from tensorflow.keras import backend as K

from tensorflow.keras import initializers, regularizers, constraints

from tensorflow.keras.layers import Layer





class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        """

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            # 1

            model.add(LSTM(64, return_sequences=True))

            model.add(Attention())

            # next add a Dense layer (for classification/regression) or whatever...

            # 2

            hidden = LSTM(64, return_sequences=True)(words)

            sentence = Attention()(hidden)

            # next add a Dense layer (for classification/regression) or whatever...

        """

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



        self.W = self.add_weight(name='{}_W'.format(self.name),

                                 shape=(input_shape[-1],),

                                 initializer=self.init,

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight(name='{}_b'.format(self.name),

                                     shape=(input_shape[1],),

                                     initializer='zero',

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)

        if self.bias:

            e += self.b

        e = K.tanh(e)



        a = K.exp(e)

        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)



        c = K.sum(a * x, axis=1)

        return c



    def compute_output_shape(self, input_shape):

        return input_shape[0], self.features_dim
from tensorflow.keras import Model

from tensorflow.keras.layers import Embedding, Dense, CuDNNLSTM, Bidirectional



class TextAttBiRNN(Model):

    def __init__(self,

                 maxlen,

                 max_features,

                 embedding_dims,

                 class_num=1,

                 last_activation='sigmoid'):

        super(TextAttBiRNN, self).__init__()

        self.maxlen = maxlen

        self.max_features = max_features

        self.embedding_dims = embedding_dims

        self.class_num = class_num

        self.last_activation = last_activation

        self.embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)

        self.bi_rnn = Bidirectional(CuDNNLSTM(128, return_sequences=True))  # LSTM or GRU

        self.attention = Attention(self.maxlen)

        self.classifier = Dense(self.class_num, activation=self.last_activation)



    def call(self, inputs):

        if len(inputs.get_shape()) != 2:

            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))

        if inputs.get_shape()[1] != self.maxlen:

            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        embedding = self.embedding(inputs)

        x = self.bi_rnn(embedding)

        x = self.attention(x)

        output = self.classifier(x)

        return output
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing import sequence



max_features = 5000

maxlen = 400

batch_size = 32

embedding_dims = 50

epochs = 10



print('Loading data...')

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')

print(len(x_test), 'test sequences')



print('Pad sequences (samples x time)...')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print('Build model...')

model = TextAttBiRNN(maxlen, max_features, embedding_dims)

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])



print('Train...')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')

model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          callbacks=[early_stopping],

          validation_data=(x_test, y_test))



print('Test...')

result = model.predict(x_test)