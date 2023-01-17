!pip uninstall -y -q allennlp preprocessing 

!pip install -q nltk --upgrade rouge
import csv

import math

import time



import gensim

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy as np

import tensorflow as tf

from nltk.translate import bleu_score, gleu_score, nist_score

from rouge import Rouge

from sklearn.model_selection import train_test_split



# enable eager execution for TensorFlow < 2.0

tf.compat.v1.enable_eager_execution()



l2 = tf.keras.regularizers.l2(0.01)
class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):

        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)

        self.W2 = tf.keras.layers.Dense(units)

        self.V = tf.keras.layers.Dense(1)



    def call(self, query, **kwargs):

        values = kwargs["values"]



        # hidden shape == (batch_size, hidden size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden size)

        # we are doing this to perform addition to calculate the score

        hidden_with_time_axis = tf.expand_dims(query, 1)



        # score shape == (batch_size, max_length, 1)

        # we get 1 at the last axis because we are applying score to self.V

        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))



        # attention_weights shape == (batch_size, max_length, 1)

        attention_weights = tf.nn.softmax(score, axis=1)



        # context_vector shape after sum == (batch_size, hidden_size)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)



        return context_vector, attention_weights
def rnn_layer(units):

    return tf.keras.layers.GRU(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                               kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                               return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = True



    def __init__(self, enc_units):

        super(Encoder, self).__init__()

        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        output, forward, backward, = self.rnn(x)



        # concatenate forward and backward

        state = tf.keras.layers.Concatenate()([forward, backward])



        return output, state





class Decoder(tf.keras.Model):

    attention_mechanism = True



    def __init__(self, vocab_size, dec_units):

        super(Decoder, self).__init__()

        self.rnn = rnn_layer(dec_units * 2)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



        # attention mechanism

        self.attention = BahdanauAttention(dec_units)



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]

        encoder_output = kwargs["encoder_output"]



        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(state, values=encoder_output)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state, attention_weights
def rnn_layer(units):

    return tf.keras.layers.LSTM(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                                kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                                return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = True



    def __init__(self, enc_units):

        super(Encoder, self).__init__()

        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        output, forward_h, forward_c, backward_h, backward_c = self.rnn(x)



        # concatenate forward and backward

        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])

        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])



        return output, [state_h, state_c]





class Decoder(tf.keras.Model):

    attention_mechanism = True



    def __init__(self, vocab_size, dec_units):

        super(Decoder, self).__init__()

        self.rnn = rnn_layer(dec_units * 2)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



        # attention mechanism

        self.attention = BahdanauAttention(dec_units)



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state_h, state_c = kwargs["state"]

        encoder_output = kwargs["encoder_output"]



        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(state_h, values=encoder_output)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state_h, state_c = self.rnn(x, initial_state=[state_h, state_c])



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, [state_h, state_c], attention_weights
def rnn_layer(units):

    return tf.keras.layers.SimpleRNN(units, recurrent_initializer="glorot_uniform", kernel_regularizer=l2,

                                     recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                                     return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = True



    def __init__(self, enc_units):

        super(Encoder, self).__init__()

        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        output, forward, backward, = self.rnn(x)



        # concatenate forward and backward

        state = tf.keras.layers.Concatenate()([forward, backward])



        return output, state





class Decoder(tf.keras.Model):

    attention_mechanism = True



    def __init__(self, vocab_size, dec_units):

        super(Decoder, self).__init__()

        self.rnn = rnn_layer(dec_units * 2)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



        # attention mechanism

        self.attention = BahdanauAttention(dec_units)



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]

        encoder_output = kwargs["encoder_output"]



        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(state, values=encoder_output)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state, attention_weights
def rnn_layer(units):

    return tf.keras.layers.GRU(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                               kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                               return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = True



    def __init__(self, enc_units):

        super(Encoder, self).__init__()

        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        _, forward, backward, = self.rnn(x)



        # concatenate forward and backward

        state = tf.keras.layers.Concatenate()([forward, backward])



        return state





class Decoder(tf.keras.Model):

    attention_mechanism = False



    def __init__(self, vocab_size, dec_units):

        super(Decoder, self).__init__()

        self.rnn = rnn_layer(dec_units * 2)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(state, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state
def rnn_layer(units):

    return tf.keras.layers.GRU(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                               kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                               return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = True



    def __init__(self, enc_units):

        super(Encoder, self).__init__()

        self.rnn = rnn_layer(enc_units)



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        output, state = self.rnn(x)



        return output, state





class Decoder(tf.keras.Model):

    attention_mechanism = True



    def __init__(self, vocab_size, dec_units):

        super(Decoder, self).__init__()

        self.rnn = rnn_layer(dec_units)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



        # attention mechanism

        self.attention = BahdanauAttention(dec_units)



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]

        encoder_output = kwargs["encoder_output"]



        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(state, values=encoder_output)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state, attention_weights
def rnn_layer(units):

    return tf.keras.layers.GRU(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                               kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                               return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = False



    def __init__(self, vocab_size, embedding_size, enc_units):

        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(enc_units, activation='tanh')



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        x = self.embedding(x)



        # flatten the x

        x = self.flatten(x)

        output = self.dense(x)



        return output





class Decoder(tf.keras.Model):

    attention_mechanism = False



    def __init__(self, vocab_size, embedding_size, dec_units):

        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        self.rnn = rnn_layer(dec_units)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



    def call(self, x, **kwargs):

        # x shape == (batch_size, 1, embedding_dim)

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]



        # x shape == (batch_size, 1, embedding_dim)

        x = self.embedding(x)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(state, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state
def rnn_layer(units):

    return tf.keras.layers.GRU(units, recurrent_activation="sigmoid", recurrent_initializer="glorot_uniform",

                               kernel_regularizer=l2, recurrent_regularizer=l2, dropout=0.5, recurrent_dropout=0.5,

                               return_sequences=True, return_state=True)





class Encoder(tf.keras.Model):

    pre_trained_word2vec = False



    def __init__(self, vocab_size, embedding_size, enc_units):

        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        self.rnn = tf.keras.layers.Bidirectional(rnn_layer(enc_units))



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        x = self.embedding(x)



        output, forward, backward, = self.rnn(x)



        # concatenate forward and backward

        state = tf.keras.layers.Concatenate()([forward, backward])



        return output, state





class Decoder(tf.keras.Model):

    attention_mechanism = True



    def __init__(self, vocab_size, embedding_size, dec_units):

        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        self.rnn = rnn_layer(dec_units * 2)

        self.fc = tf.keras.layers.Dense(vocab_size, activation="softmax")



        # attention mechanism

        self.attention = BahdanauAttention(dec_units)



    def call(self, x, **kwargs):

        x = tf.cast(x, dtype=tf.float32)



        state = kwargs["state"]

        encoder_output = kwargs["encoder_output"]



        # x shape == (batch_size, 1, embedding_dim)

        x = self.embedding(x)



        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(state, values=encoder_output)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the rnn layer

        output, state = self.rnn(x, initial_state=state)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state, attention_weights
# data_15 means threshold value is 1.5

path_to_file = "../input/toplab/data_15.csv"



# word embedding size

embedding_size = 300



# True for applying the early stopping

early_stopping = True



# True for applying the attention mechanism

decoder_attention = Decoder.attention_mechanism



# True for applying the pre-trained word2vec

pre_trained_word2vec = Encoder.pre_trained_word2vec
# load the word2vec model

if pre_trained_word2vec and "model" not in locals():

    model = gensim.models.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

    vocab = model.vocab



    embedding_size = 300



    token_index = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}



    # word vectors for tokens

    token_vector = {"<start>": tf.ones(embedding_size),

                    "<end>": tf.negative(tf.ones(embedding_size)),

                    "<unk>": tf.zeros(embedding_size),

                    "<pad>": tf.tile([0.5], [embedding_size])}
# function for pre-processing the sentences

def preprocess_sentence(sent):

    # Google pre-trained word2vec model uses _ instead of -

    sent = sent.replace("-", "_")



    if pre_trained_word2vec:

        return "<start> " + " ".join(topic if topic in vocab else "<unk>" for topic in sent.split()) + " <end>"

    return "<start> " + sent + " <end>"





# function for loading and split the dataset

def create_dataset(path):

    topics = []

    labels = []



    with open(path, "r") as csv_data:

        reader = csv.reader(csv_data)



        # skip header

        next(reader, None)



        for row in reader:

            topic_str = preprocess_sentence(row[0])

            label_str = preprocess_sentence(row[1])



            if pre_trained_word2vec:

                if all(label in token_vector for label in label_str.split()):

                    continue



            topics.append(topic_str)

            labels.append(label_str)



    return topics, labels





# max sentence length

def max_length(vectors):

    return max(len(vector) for vector in vectors)





# tokenize the sentence

def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")

    lang_tokenizer.fit_on_texts(lang)



    indices_list = lang_tokenizer.texts_to_sequences(lang)



    indices_list = tf.keras.preprocessing.sequence.pad_sequences(indices_list, padding="post")



    return indices_list, lang_tokenizer





# {topic: [label_1, label_2, ...]}

def create_reference_dict(inputs, targets):

    ref_dict = {}



    for topic, label in zip(inputs, targets):

        if topic not in ref_dict:

            ref_dict[topic] = []

        ref_dict[topic].append(label)



    return ref_dict





# convert word index to vector

def index2vec(index, tokenizer):

    if index <= 3:

        return token_vector[token_index[index]]

    return model.word_vec(tokenizer.index_word[index])





# convert a list of indices to vectors

def indices2vec(indices, tokenizer):

    return [index2vec(int(index), tokenizer) for index in indices]





# input sequence to input & target vectors

def input2vec(data):

    inputs = []

    targets = []



    for input_seq in data:

        for target_seq in reference_dict[input_seq]:

            inputs.append(input_seq)

            targets.append(target_seq)



    inputs = input_tokenizer.texts_to_sequences(inputs)

    targets = target_tokenizer.texts_to_sequences(targets)



    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length_inp, padding="post")

    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=max_length_target, padding="post")



    return inputs, targets





# creating cleaned input, output pairs

input_lang, target_lang = create_dataset(path_to_file)



reference_dict = create_reference_dict(input_lang, target_lang)



input_vectors, input_tokenizer = tokenize(input_lang)

target_vectors, target_tokenizer = tokenize(target_lang)



# calculate max_length of the vectors

max_length_inp, max_length_target = max_length(input_vectors), max_length(target_vectors)



# creating training, val, test sets using an 70-20-10 split

input_train, input_test = train_test_split(list(reference_dict.keys()), test_size=0.3)

input_val, input_test = train_test_split(input_test, test_size=0.33)



train_vocab = set([word for sentence in input_train for word in sentence.split()])

test_vocab = set([word for sentence in input_test for word in sentence.split()])

intersect_vocab = train_vocab.intersection(test_vocab)

print("%.2f%% of words in the test set are unknown" % ((1 - len(intersect_vocab) / len(test_vocab)) * 100))



input_train, target_train = input2vec(input_train)

input_val, target_val = input2vec(input_val)

input_test, target_test = input2vec(input_test)



BATCH_SIZE = 64

buffer_size = len(input_train)

vocab_inp_size = len(input_tokenizer.word_index) + 1

vocab_tar_size = len(target_tokenizer.word_index) + 1

train_steps_per_epoch = math.ceil(len(input_train) / BATCH_SIZE)

val_steps_per_epoch = math.ceil(len(input_val) / BATCH_SIZE)

test_steps_per_epoch = math.ceil(len(input_test) / BATCH_SIZE)



# train dataset

train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))

train_dataset = train_dataset.shuffle(buffer_size).batch(BATCH_SIZE)



# validation dataset

val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))

val_dataset = val_dataset.batch(BATCH_SIZE)



# test dataset

test_dataset = tf.data.Dataset.from_tensor_slices((input_test, target_test))

test_dataset = test_dataset.batch(BATCH_SIZE)
# RNN units dimension

RNN_DIMENSION = 1024



# initialise encoder decoder with pre_trained_word2vec flag

if pre_trained_word2vec:

    encoder, decoder = Encoder(RNN_DIMENSION), Decoder(vocab_tar_size, RNN_DIMENSION)

else:

    encoder = Encoder(vocab_inp_size, embedding_size, RNN_DIMENSION)

    decoder = Decoder(vocab_tar_size, embedding_size, RNN_DIMENSION)



train_l = []

train_acc = []

val_l = []

val_acc = []

bleu_scores = []

gleu_scores = []

nist_scores = []

rouge_1l_dicts = []





# custom learning rate scheduler

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):



    def __init__(self, warm_up_steps=2000):

        super(CustomSchedule, self).__init__()



        self.warm_up_steps = warm_up_steps



        self.d_model = int(4000 / warm_up_steps) * 128

        self.d_model = tf.cast(self.d_model, tf.float32)



    def get_config(self):

        pass



    def __call__(self, step):

        arg1 = tf.math.rsqrt(step)

        arg2 = step * (self.warm_up_steps ** -1.5)



        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)





learning_rate = CustomSchedule()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")





# loss function

def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)



    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask



    return tf.reduce_mean(loss_)





# loss & accuracy for training

train_loss = tf.keras.metrics.Mean(name="train_loss")

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")



# loss & accuracy for validation and testing

test_loss = tf.keras.metrics.Mean(name="test_loss")

test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")





# training function

def train_step(inputs, targets):

    loss = 0

    enc_output = None



    with tf.GradientTape() as tape:



        if pre_trained_word2vec:

            inputs = [indices2vec(indices, input_tokenizer) for indices in inputs]

            

        if decoder_attention:

            enc_output, enc_hidden = encoder(inputs)

        else:

            enc_hidden = encoder(inputs)



        dec_hidden = enc_hidden



        if pre_trained_word2vec:

            dec_input = tf.expand_dims(tf.expand_dims(token_vector["<start>"], 0), 0)

            dec_input = tf.tile(dec_input, [targets.shape[0], 1, 1])

        else:

            dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]] * targets.shape[0], 1)

            

        # teacher forcing - feeding the target as the next input

        for t in range(1, targets.shape[1]):

            # passing enc_output to the decoder

            if decoder_attention:

                predictions, dec_hidden, _ = decoder(dec_input, state=dec_hidden, encoder_output=enc_output)

            else:

                predictions, dec_hidden = decoder(dec_input, state=dec_hidden)



            loss += loss_function(targets[:, t], predictions)

            train_accuracy.update_state(targets[:, t], predictions)



            # using teacher forcing

            if pre_trained_word2vec:

                dec_input = tf.expand_dims(indices2vec(targets[:, t], target_tokenizer), 1)

            else:

                dec_input = tf.expand_dims(targets[:, t], 1)



    train_loss((loss / int(targets.shape[1])))



    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))





# validation & testing function

def test_step(inputs, targets):

    loss = 0

    enc_output = None



    if pre_trained_word2vec:

        inputs = [indices2vec(indices, input_tokenizer) for indices in inputs]



    if decoder_attention:

        enc_output, enc_hidden = encoder(inputs)

    else:

        enc_hidden = encoder(inputs)



    dec_hidden = enc_hidden



    predicted_labels = []



    if pre_trained_word2vec:

        dec_input = tf.expand_dims(tf.expand_dims(token_vector["<start>"], 0), 0)

        dec_input = tf.tile(dec_input, [targets.shape[0], 1, 1])

    else:

        dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]] * targets.shape[0], 1)



    for t in range(1, max_length_target):

        # passing enc_output to the decoder

        if decoder_attention:

            predictions, dec_hidden, _ = decoder(dec_input, state=dec_hidden, encoder_output=enc_output)

        else:

            predictions, dec_hidden = decoder(dec_input, state=dec_hidden)



        loss += loss_function(targets[:, t], predictions)

        test_accuracy.update_state(targets[:, t], predictions)



        predicted = tf.math.argmax(predictions, axis=1)

        predicted_labels.append(list(predicted.numpy()))



        if pre_trained_word2vec:

            dec_input = tf.expand_dims(indices2vec(predicted, target_tokenizer), 1)

        else:

            dec_input = tf.expand_dims(predicted, 1)



    test_loss((loss / int(targets.shape[1])))



    # result for evaluation

    result = []



    # rotate the predicted matrix

    for label in zip(*predicted_labels):

        result.append([])

        for value in label:

            if value in (0, 2):

                break

            result[-1].append(value)



    return result





# split input and remove <start> & <end> tokens

def word_split(sent):

    return [label.split()[1:-1] for label in reference_dict[sent]]





# sum the rouge score

def rouge_sum_score(rouge_dict):

    return sum(value for fpr in rouge_dict.values() for value in fpr.values())





# format the rouge dictionary for output

def rouge_dict_format(rouge_dict):

    return "{rouge-1: {f: %f, p: %f, r: %f}, rouge-l: {f: %f, p: %f, r: %f}}" % (

        rouge_dict["rouge-1"]["f"], rouge_dict["rouge-1"]["p"], rouge_dict["rouge-1"]["r"],

        rouge_dict["rouge-l"]["f"], rouge_dict["rouge-l"]["p"], rouge_dict["rouge-l"]["r"])





# BLEU-1, GLEU-1, NIST-1, ROUGE-1 & ROUGE-L evaluation metrics

def evaluation_metrics(dataset, steps, size):

    references = []

    hypotheses = []



    rouge = Rouge()

    rouge_dict = {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},

                  "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},

                  "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}



    # make references & hypotheses lists

    for inputs, targets in dataset.take(steps):

        for labels in target_tokenizer.sequences_to_texts(test_step(inputs, targets)):

            if len(labels) > 0:

                hypotheses.append(labels.split())

            else:

                hypotheses.append([""])



        for labels in input_tokenizer.sequences_to_texts(inputs.numpy()):

            references.append(word_split(labels))



    for index, hypothesis in enumerate(hypotheses):

        max_score = {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},

                     "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},

                     "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}



        # one hypothesis may have several references

        for reference in references[index]:

            try:

                rouge_score = rouge.get_scores(" ".join(hypothesis), " ".join(reference))[0]

                # keep the best score

                if rouge_sum_score(rouge_score) > rouge_sum_score(max_score):

                    max_score = rouge_score

            except ValueError:

                pass



        for method_key in rouge_dict:

            # fpr for traversing f1 precision recall

            for fpr in rouge_dict[method_key]:

                rouge_dict[method_key][fpr] += max_score[method_key][fpr]



    # average

    for method_key in rouge_dict:

        for fpr in rouge_dict[method_key]:

            rouge_dict[method_key][fpr] /= size



    bleu = bleu_score.corpus_bleu(references, hypotheses, weights=(1,))

    gleu = gleu_score.corpus_gleu(references, hypotheses, max_len=1)

    nist = nist_score.corpus_nist(references, hypotheses, n=1)



    print("BLEU-1 Score: %.4f" % bleu)

    print("GLEU-1 Score: %.4f" % gleu)

    print("NIST-1 Score: %.4f" % nist)

    print("ROUGE Scores: %s" % rouge_dict_format(rouge_dict))



    return bleu, gleu, nist, rouge_dict





# plot a single subplot

def single_plot(ax, epochs, data, title):

    ax.plot(epochs, data)

    ax.set_xlabel("Epoch")

    ax.set_title(title)





# plot loss, accuracy, BLEU-1, GLEU-1, NIST-1, ROUGE-1, ROUGE-L result

def plot_result(t_l, t_acc, v_l, v_acc, bleu, gleu, nist, rouge_1l):

    epochs = list(range(1, len(t_l) + 1))

    plt.figure(figsize=(16, 16))



    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=3)

    ax1.plot(epochs, t_l, label="Train Loss")

    ax1.plot(epochs, v_l, label="Valid Loss")

    ax1.legend()

    ax1.set_ylim([0, 4])

    ax1.set_xlabel("Epoch")

    ax1.set_title("Loss")



    ax2 = plt.subplot2grid((4, 6), (0, 3), colspan=3)

    ax2.plot(epochs, t_acc, label="Train Accuracy")

    ax2.plot(epochs, v_acc, label="Valid Accuracy")

    ax2.legend()

    ax2.set_ylim([0, 1])

    ax2.set_xlabel("Epoch")

    ax2.set_title("Accuracy")



    rouge_1_f = [rouge_dict["rouge-1"]["f"] for rouge_dict in rouge_1l]

    rouge_1_p = [rouge_dict["rouge-1"]["p"] for rouge_dict in rouge_1l]

    rouge_1_r = [rouge_dict["rouge-1"]["r"] for rouge_dict in rouge_1l]

    rouge_l_f = [rouge_dict["rouge-l"]["f"] for rouge_dict in rouge_1l]

    rouge_l_p = [rouge_dict["rouge-l"]["p"] for rouge_dict in rouge_1l]

    rouge_l_r = [rouge_dict["rouge-l"]["r"] for rouge_dict in rouge_1l]



    ax3 = plt.subplot2grid((4, 6), (1, 0), colspan=2)

    ax4 = plt.subplot2grid((4, 6), (1, 2), colspan=2)

    ax5 = plt.subplot2grid((4, 6), (1, 4), colspan=2)

    ax6 = plt.subplot2grid((4, 6), (2, 0), colspan=2)

    ax7 = plt.subplot2grid((4, 6), (2, 2), colspan=2)

    ax8 = plt.subplot2grid((4, 6), (2, 4), colspan=2)

    ax9 = plt.subplot2grid((4, 6), (3, 0), colspan=2)

    ax10 = plt.subplot2grid((4, 6), (3, 2), colspan=2)

    ax11 = plt.subplot2grid((4, 6), (3, 4), colspan=2)

    

    print("validation", max(v_acc), v_acc[-1])

    print("bleu-1", max(bleu), bleu[-1])

    print("gleu-1", max(gleu), gleu[-1])

    print("nist-1", max(nist), nist[-1])

    print("rouge_1_f", max(rouge_1_f), rouge_1_f[-1])

    print("rouge_l_f", max(rouge_l_f), rouge_l_f[-1])



    single_plot(ax3, epochs, bleu, "BLEU-1 Scores")

    single_plot(ax4, epochs, gleu, "GLEU-1 Scores")

    single_plot(ax5, epochs, nist, "NIST-1 Scores")

    single_plot(ax6, epochs, rouge_1_f, "ROUGE-1 F1")

    single_plot(ax7, epochs, rouge_1_p, "ROUGE-1 Precision")

    single_plot(ax8, epochs, rouge_1_r, "ROUGE-1 Recall")

    single_plot(ax9, epochs, rouge_l_f, "ROUGE-L F1")

    single_plot(ax10, epochs, rouge_l_p, "ROUGE-L Precision")

    single_plot(ax11, epochs, rouge_l_r, "ROUGE-L Recall")



    plt.tight_layout()

    plt.show()
EPOCH = 50

PATIENCE = 6

stop_flags = []

last_loss = float("inf")



for epoch in range(EPOCH):

    start = time.time()



    # reset the loss and accuracy each epoch

    train_loss.reset_states()

    train_accuracy.reset_states()



    test_loss.reset_states()

    test_accuracy.reset_states()



    # shuffle the dataset each epoch

    train_dataset = train_dataset.shuffle(buffer_size)



    for inp, target in train_dataset.take(train_steps_per_epoch):

        train_step(inp, target)



    bleu_1, gleu_1, nist_1, rouge_1l_dict = evaluation_metrics(val_dataset, val_steps_per_epoch, len(input_val))



    print("Train Loss: %.4f Accuracy: %.4f" % (train_loss.result(), train_accuracy.result()))

    print("Validation Loss: %.4f Accuracy: %.4f" % (test_loss.result(), test_accuracy.result()))

    print("%.4f secs taken for epoch %d\n" % (time.time() - start, epoch + 1))



    train_l.append(train_loss.result().numpy())

    train_acc.append(train_accuracy.result().numpy())

    val_l.append(test_loss.result().numpy())

    val_acc.append(test_accuracy.result().numpy())

    bleu_scores.append(bleu_1)

    gleu_scores.append(gleu_1)

    nist_scores.append(nist_1)

    rouge_1l_dicts.append(rouge_1l_dict)



    # early stopping

    val_loss = test_loss.result().numpy()



    if early_stopping:

        if val_loss > last_loss or abs(val_loss - last_loss) < 1e-3:

            stop_flags.append(True)

        else:

            stop_flags.clear()



        # score is continuously decreasing

        if len(stop_flags) >= PATIENCE:

            print("\nEarly stopping\n")

            break



        last_loss = val_loss



# plot result

plot_result(train_l, train_acc, val_l, val_acc, bleu_scores, gleu_scores, nist_scores, rouge_1l_dicts)



# reset the loss and accuracy

test_loss.reset_states()

test_accuracy.reset_states()



evaluation_metrics(test_dataset, test_steps_per_epoch, len(input_test))



print("Test Loss: %.4f Accuracy: %.4f\n" % (test_loss.result(), test_accuracy.result()))
# function for plotting the attention weights

def plot_attention(attention, result, sentence):

    result = result.split()

    sentence = sentence.split()

    attention = attention[:len(result), :len(sentence)]



    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap="viridis")



    font_dict = {"fontsize": 14}



    ax.set_xticklabels([""] + sentence, fontdict=font_dict, rotation=90)

    ax.set_yticklabels([""] + result, fontdict=font_dict)



    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



    plt.show()





# test the samples

def sample_test(sentence):

    result = ""

    enc_out = None

    sentence = preprocess_sentence(sentence)

    attention_plot = np.zeros((max_length_target, max_length_inp))



    inputs = [input_tokenizer.word_index[w] for w in sentence.split(" ")]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")



    if pre_trained_word2vec:

        inputs = [indices2vec(indices, input_tokenizer) for indices in inputs]

    else:

        inputs = tf.convert_to_tensor(inputs)



    if decoder_attention:

        enc_out, enc_hidden = encoder(inputs)

    else:

        enc_hidden = encoder(inputs)



    dec_hidden = enc_hidden



    if pre_trained_word2vec:

        dec_input = tf.expand_dims(tf.expand_dims(token_vector["<start>"], 0), 0)

    else:

        dec_input = tf.expand_dims([target_tokenizer.word_index["<start>"]], 0)



    for t in range(max_length_target):



        if decoder_attention:

            predictions, dec_hidden, attention_weights = decoder(dec_input, state=dec_hidden, encoder_output=enc_out)



            # storing the attention weights to plot later on

            attention_weights = tf.reshape(attention_weights, (-1,))

            attention_plot[t] = attention_weights.numpy()

        else:

            predictions, dec_hidden = decoder(dec_input, state=dec_hidden)



        predicted_id = tf.math.argmax(predictions[0]).numpy()

        result += target_tokenizer.index_word[predicted_id] + " "



        # stop for reaching the end of the predicted sequence

        if target_tokenizer.index_word[predicted_id] == "<end>":

            break



        # the predicted ID is fed back into the model

        if pre_trained_word2vec:

            dec_input = tf.expand_dims([index2vec(predicted_id, target_tokenizer)], 1)

        else:

            dec_input = tf.expand_dims([predicted_id], 0)



    if decoder_attention:

        result = result.strip()

        plot_attention(attention_plot, result, sentence)



    return result, sentence





# output the predicted label

def generate_label(sentence):

    result, sentence = sample_test(sentence)



    print("Input labels: %s" % sentence)

    print("Predicted topic: %s" % "<start> " + result)

    # output the references

    if sentence in reference_dict:

        print("Target topic: %s\n" % ', '.join(reference_dict[sentence]))
# sample test from the test dataset

test_iter = test_dataset.make_one_shot_iterator()

sample = test_iter.get_next()[0]



for i in [0, 20, 40, 60]:

    text = input_tokenizer.sequences_to_texts([sample[i].numpy()[1:-1]])[0]

    generate_label(text)



# sample test from the train dataset

train_iter = train_dataset.make_one_shot_iterator()

sample = test_iter.get_next()[0]



for i in [0, 20]:

    text = input_tokenizer.sequences_to_texts([sample[i].numpy()[1:-1]])[0]

    generate_label(text)