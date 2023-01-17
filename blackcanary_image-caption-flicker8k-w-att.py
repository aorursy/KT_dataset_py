# Imports



# General

import numpy as np 

import pandas as pd 

import string

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pylab import imread, subplot, imshow, show

from pickle import dump

from tqdm import tqdm

import os



# Tokenizer

from keras.preprocessing.text import Tokenizer



# Converting image to vector of 4,096 elements

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model



# Attention model

import tensorflow as tf

from keras import backend as K

from keras import regularizers, constraints, initializers, activations

from keras.layers.recurrent import Recurrent

from keras.engine import InputSpec



# Creating our model

from pickle import load

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.utils import plot_model

from keras.models import load_model

from keras.models import Model

from keras.models import Sequential

from keras.layers import Input

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

from keras.layers import Dropout

from keras.layers.merge import add

from keras.callbacks import ModelCheckpoint

from IPython.display import display, Image                                                                                



from nltk.translate.bleu_score import corpus_bleu
print(os.listdir("../input"))
# Preparing the text dataset for storing the captions with indices as clean descriptions

# and also grabbing the vocabulary from the captions and storing all the words in the vocabulary



def load_descriptions(filename):

    file = open(filename, 'r')

    doc = file.read()

    file.close()



    mapping = dict()

    for line in doc.split('\n'):

        tokens = line.split()

        if len(line) < 2:

            continue



        image_id, image_desc = tokens[0], tokens[1:]

        image_id = image_id.split('.')[0]

        image_desc = ' '.join(image_desc)



        if image_id not in mapping:

            mapping[image_id] = list()



        mapping[image_id].append(image_desc)



    return mapping



def clean_descriptions(descriptions):

    table = str.maketrans('', '', string.punctuation)

    for key, desc_list in descriptions.items():

        for i in range(len(desc_list)):

            desc = desc_list[i]

            desc = desc.split()

            desc = [word.lower() for word in desc]

            desc = [w.translate(table) for w in desc]

            desc = [word for word in desc if len(word)>1]

            desc = [word for word in desc if word.isalpha()]

            desc_list[i] = ' '.join(desc)



def to_vocabulary(descriptions):

    vocab = set()

    for key in descriptions.keys():

        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab



def save_vocabulary(vocab, filename):

    file = open(filename, 'w')

    for word in vocab:

        file.write(word+"\n")

    file.close()



def save_descriptions(descriptions, filename):

    lines = list()

    for key, desc_list in descriptions.items():

        for desc in desc_list:

            lines.append(key + ' ' + desc)



    data = '\n'.join(lines)

    file = open(filename, 'w')

    file.write(data)

    file.close()



filename = "../input/flicker8k-dataset/flickr8k_text/Flickr8k.token.txt"

descriptions = load_descriptions(filename)

print("Loaded: ", len(descriptions))



clean_descriptions(descriptions)

vocabulary = to_vocabulary(descriptions)

print("Vocabulary size: ", len(vocabulary))



save_vocabulary(vocabulary, "vocabulary.txt")

save_descriptions(descriptions, "descriptions.txt")
# tokenizer



def load_doc(filename):

    file = open(filename, 'r')

    text = file.read()

    file.close()

    return text



def load_set(filename):

    doc = load_doc(filename)

    dataset = list()

    for line in doc.split('\n'):

        if len(line) < 1:

            continue

        identifier = line.split('.')[0]

        dataset.append(identifier)

    return set(dataset)



def load_clean_descriptions(filename, dataset):

    doc = load_doc(filename)

    descriptions = dict()

    for line in doc.split('\n'):

        tokens = line.split()

        image_id, image_desc = tokens[0], tokens[1:]

        if image_id in dataset:

            if image_id not in descriptions:

                descriptions[image_id] = list()

            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            descriptions[image_id].append(desc)

    return descriptions



def to_lines(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc



def create_tokenizer(descriptions):

    lines = to_lines(descriptions)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



filename = "../input/flicker8k-dataset/flickr8k_text/Flickr_8k.trainImages.txt"

train = load_set(filename)

print('Dataset: %d' % len(train))



train_descriptions = load_clean_descriptions('descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))



tokenizer = create_tokenizer(train_descriptions)



dump(tokenizer, open('tokenizer.pkl', 'wb'))
# Using the VGG16 network to extract features from each image. 

# The resulting feature file will have features of each image as a 1D, 4096 element array

# Defining a function to extract features from images stored in a given directory



def extract(dirname):



    model = VGG16()

    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    print(model.summary())



    features = dict()



    for name in tqdm(os.listdir(dirname)):

        filename = dirname + '/' + name

        image = load_img(filename, target_size=(224, 224))

        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        image_id = name.split('.')[0]

        features[image_id] = feature



    return features



directory_name = "../input/flicker8k-dataset/flickr8k_dataset/Flicker8k_Dataset"

features = extract(directory_name)



print("Extracted Features: ", len(features))



# dumping features into a pickled file

dump(features, open('features.pkl', 'wb'))
def time_distributed_dense(x, w, b=None, dropout=None, input_dim=None, output_dim=None, timesteps=None):

        '''Apply y.w + b for every temporal slice y of x.

        '''

        if not input_dim:

            input_dim = K.shape(x)[2]

        if not timesteps:

            timesteps = K.shape(x)[1]

        if not output_dim:

            output_dim = K.shape(w)[1]



        if dropout:

            ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))

            dropout_matrix = K.dropout(ones, dropout)

            expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)

            x *= expanded_dropout_matrix



        x = K.reshape(x, (-1, input_dim))



        x = K.dot(x, w)

        if b:

            x = x + b

        x = K.reshape(x, (-1, timesteps, output_dim))

        return x
# Attention layer



tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)



class AttentionDecoder(Recurrent):



    def __init__(self, units, output_dim, activation='tanh', return_probabilities=False, name='AttentionDecoder', kernel_initializer='glorot_uniform', 

                 recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 

                 kernel_constraint=None, bias_constraint=None, **kwargs):

        """

        Implements an AttentionDecoder that takes in a sequence encoded by an

        encoder and outputs the decoded states

        :param units: dimension of the hidden state and the attention matrices

        :param output_dim: the number of labels in the output space



        """

        self.units = units

        self.output_dim = output_dim

        self.return_probabilities = return_probabilities

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)

        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.bias_initializer = initializers.get(bias_initializer)



        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.recurrent_regularizer = regularizers.get(kernel_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)



        self.kernel_constraint = constraints.get(kernel_constraint)

        self.recurrent_constraint = constraints.get(kernel_constraint)

        self.bias_constraint = constraints.get(bias_constraint)



        super(AttentionDecoder, self).__init__(**kwargs)

        self.name = name

        self.return_sequences = True  # must return sequences



    def build(self, input_shape):



        self.batch_size, self.timesteps, self.input_dim = input_shape



        if self.stateful:

            super(AttentionDecoder, self).reset_states()



        self.states = [None, None]  # y, s



        """

            Matrices for creating the context vector

        """



        self.V_a = self.add_weight(shape=(self.units,), name='V_a', initializer=self.kernel_initializer,

                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        self.U_a = self.add_weight(shape=(self.input_dim, self.units), name='U_a', initializer=self.kernel_initializer,

                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        self.W_a = self.add_weight(shape=(self.units, self.units), name='W_a', initializer=self.kernel_initializer,

                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

        self.b_a = self.add_weight(shape=(self.units,), name='b_a', initializer=self.bias_initializer,

                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        """

            Matrices for the r (reset) gate

        """

        self.C_r = self.add_weight(shape=(self.input_dim, self.units), name='C_r', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.U_r = self.add_weight(shape=(self.units, self.units), name='U_r', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.W_r = self.add_weight(shape=(self.output_dim, self.units), name='W_r', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.b_r = self.add_weight(shape=(self.units, ), name='b_r', initializer=self.bias_initializer,

                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint)



        """

            Matrices for the z (update) gate

        """

        self.C_z = self.add_weight(shape=(self.input_dim, self.units), name='C_z', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.U_z = self.add_weight(shape=(self.units, self.units), name='U_z', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.W_z = self.add_weight(shape=(self.output_dim, self.units), name='W_z', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.b_z = self.add_weight(shape=(self.units, ), name='b_z', initializer=self.bias_initializer,

                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        """

            Matrices for the proposal

        """

        self.C_p = self.add_weight(shape=(self.input_dim, self.units), name='C_p', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.U_p = self.add_weight(shape=(self.units, self.units), name='U_p', initializer=self.recurrent_initializer, 

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.W_p = self.add_weight(shape=(self.output_dim, self.units), name='W_p', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.b_p = self.add_weight(shape=(self.units, ), name='b_p', initializer=self.bias_initializer,

                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        """

            Matrices for making the final prediction vector

        """

        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim), name='C_o', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.U_o = self.add_weight(shape=(self.units, self.output_dim), name='U_o', initializer=self.recurrent_initializer,

                                    regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim), name='W_o',initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)

        self.b_o = self.add_weight(shape=(self.output_dim, ), name='b_o', initializer=self.bias_initializer,

                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint)



        # For creating the initial state:

        self.W_s = self.add_weight(shape=(self.input_dim, self.units), name='W_s', initializer=self.recurrent_initializer,

                                   regularizer=self.recurrent_regularizer, constraint=self.recurrent_constraint)



        self.input_spec = [

            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]

        self.built = True



    def call(self, x):

        # store the whole sequence so we can "attend" to it at each timestep

        self.x_seq = x



        # apply the a dense layer over the time dimension of the sequence

        # do it here because it doesn't depend on any previous steps

        # to save computation time:

        self._uxpb = time_distributed_dense(self.x_seq, self.U_a, b=self.b_a, input_dim=self.input_dim, timesteps=self.timesteps, output_dim=self.units)



        return super(AttentionDecoder, self).call(x)



    def get_initial_state(self, inputs):

        # apply the matrix on the first time step to get the initial s0.

        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))



        # from keras.layers.recurrent to initialize a vector of (batchsize,

        # output_dim)

        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)

        y0 = K.sum(y0, axis=(1, 2))  # (samples, )

        y0 = K.expand_dims(y0)  # (samples, 1)

        y0 = K.tile(y0, [1, self.output_dim])



        return [y0, s0]



    def step(self, x, states):



        ytm, stm = states



        # repeat the hidden state to the length of the sequence

        _stm = K.repeat(stm, self.timesteps)



        # now multiplty the weight matrix with the repeated hidden state

        _Wxstm = K.dot(_stm, self.W_a)



        # calculate the attention probabilities

        # this relates how much other timesteps contributed to this one.

        et = K.dot(activations.tanh(_Wxstm + self._uxpb), K.expand_dims(self.V_a))

        at = K.exp(et)

        at_sum = K.sum(at, axis=1)

        at_sum_repeated = K.repeat(at_sum, self.timesteps)

        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)



        # calculate the context vector

        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)

        # ~~~> calculate new hidden state

        # first calculate the "r" gate:



        rt = activations.sigmoid(K.dot(ytm, self.W_r) + K.dot(stm, self.U_r)+ K.dot(context, self.C_r) + self.b_r)



        # now calculate the "z" gate

        zt = activations.sigmoid(K.dot(ytm, self.W_z) + K.dot(stm, self.U_z) + K.dot(context, self.C_z) + self.b_z)



        # calculate the proposal hidden state:

        s_tp = activations.tanh(K.dot(ytm, self.W_p)+ K.dot((rt * stm), self.U_p) + K.dot(context, self.C_p) + self.b_p)



        # new hidden state:

        st = (1-zt)*stm + zt * s_tp



        yt = activations.softmax(K.dot(ytm, self.W_o) + K.dot(stm, self.U_o) + K.dot(context, self.C_o) + self.b_o)



        if self.return_probabilities:

            return at, [yt, st]

        else:

            return yt, [yt, st]



    def compute_output_shape(self, input_shape):

        """

            For Keras internal compatability checking

        """

        if self.return_probabilities:

            return (None, self.timesteps, self.timesteps)

        else:

            return (None, self.timesteps, self.output_dim)



    def get_config(self):

        """

            For rebuilding models on load time.

        """

        config = {

            'output_dim': self.output_dim,

            'units': self.units,

            'return_probabilities': self.return_probabilities

        }

        base_config = super(AttentionDecoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
#Train model





def load_photo_features(filename, dataset):

    all_features = load(open(filename, 'rb'))

    features = {k: all_features[k] for k in dataset}

    return features





def max_length(descriptions):

    lines = to_lines(descriptions)

    return max(len(d.split()) for d in lines)



def create_sequences(tokenizer, max_length, desc_list, photo):

    X1, X2, y = list(), list(), list()

    for desc in desc_list:

        seq = tokenizer.texts_to_sequences([desc])[0]

        for i in range(1, len(seq)):

            in_seq, out_seq = seq[:i], seq[i]

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

            X1.append(photo)

            X2.append(in_seq)

            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)



def define_model(vocab_size, max_length):

    # feature extractor model

    inputs1 = Input(shape=(4096,))

    fe1 = Dropout(0.5)(inputs1)

    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model

    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)

    se2 = Dropout(0.5)(se1)

    se3 = LSTM(256)(se2)

    #se4 = AttentionDecoder(se3)

    # decoder model

    decoder1 = add([fe2, se3])

    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.summary()



    # compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model

    model.summary()

    #plot_model(model, to_file='model.png', show_shapes=True)

    return model

    """

     inputs = Input(shape=(max_length,))

    embedding = Embedding(vocab_size, 512, mask_zero=True)(inputs)



    embed_dropout = Dropout(0.5)(embedding)



    model.add(LSTM(512, input_shape=(max_length, vocab_size), return_sequences=True))

    #model.summary()



    att = AttentionDecoder(512, max_length)(embedding)



    lstm = LSTM(512)(att)



    fc = Dense(512, activation='relu')(lstm)



    output = Dense(vocab_size, activation='softmax')(fc)



    model = Model(inputs = inputs, outputs = output)



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    model.summary()

    plot_model(model, to_file='model_custom.png', show_shapes=True)

    return model

    """



def data_generator(descriptions, photos, tokenizer, max_length):

    while 1:

        for key, desc_list in descriptions.items():

            photo = photos[key][0]

            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)

            yield [[in_img, in_seq], out_word]



filename = '../input/flicker8k-dataset/flickr8k_text/Flickr_8k.trainImages.txt'

train = load_set(filename)

print('Dataset: ', len(train))

train_descriptions = load_clean_descriptions('descriptions.txt', train)

print('Descriptions: train = ', len(train_descriptions))

train_features = load_photo_features('features.pkl', train)

print('Photos: train = ', len(train_features))

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: ', vocab_size)

max_length = max_length(train_descriptions)

print('Description Length: ', max_length)



model = define_model(vocab_size, max_length)

epochs = 20

steps = len(train_descriptions)

for i in range(epochs):

    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)

    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save('model_' + str(i) + '.h5')
#Evaluate the model



def word_for_id(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:

            return word

    return None



def generate_desc(model, tokenizer, photo, max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo,sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)

        if word is None:

            break

        in_text += ' ' + word

        if word == 'endseq':

            break

    return in_text



def evaluate_model(model, descriptions, photos, tokenizer, max_length):

    actual, predicted = list(), list()

    for key, desc_list in descriptions.items():

        yhat = generate_desc(model, tokenizer, photos[key], max_length)

        references = [d.split() for d in desc_list]

        actual.append(references)

        predicted.append(yhat.split())

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))

    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))

    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

"""



# prepare tokenizer on train set



filename = '../input/flickr8k_text/Flickr_8k.trainImages.txt'

train = load_set(filename)

print('Dataset: %d' % len(train))

train_descriptions = load_clean_descriptions('descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

max_length = max_length(train_descriptions)

print('Description Length: %d' % max_length)

"""



filename = '../input/flicker8k-dataset/flickr8k_text/Flickr_8k.testImages.txt'

test = load_set(filename)

print('Dataset: %d' % len(test))

test_descriptions = load_clean_descriptions('descriptions.txt', test)

print('Descriptions: test=%d' % len(test_descriptions))

test_features = load_photo_features('features.pkl', test)

print('Photos: test=%d' % len(test_features))



# load the model

filename = 'model_19.h5' 

model = load_model(filename)

# evaluate model

evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
#Prediction



def extract_features(filename):

    model = VGG16()

    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    image = load_img(filename, target_size=(224, 224))

    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)

    return feature





def generate_desc(model, tokenizer, photo, max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo,sequence], verbose=0)

        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)

        if word is None:

            break

        in_text += ' ' + word

        if word == 'endseq':

            break

    return in_text



tokenizer = load(open('tokenizer.pkl', 'rb'))

max_length = 34

model_file = 'model_19.h5'

model = load_model(model_file)





folder = os.listdir('../input/my-test-image/')

for img in folder:

    img_path = '../input/my-test-image/'+img

    img_features = extract_features(img_path)

    display(Image(filename=img_path))

    

    description = generate_desc(model, tokenizer, img_features, max_length)

    print(description)



"""

img_file = '../input/my-test-image/16.jpeg'

photo = extract_features(img_file)



display(Image(filename='../input/my-test-image/16.jpeg'))



description = generate_desc(model, tokenizer, photo, max_length)

print(description)



"""