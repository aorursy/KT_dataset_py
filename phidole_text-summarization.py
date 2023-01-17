from keras.models import Model
from keras.layers import Embedding, Dense, Input, RepeatVector, concatenate, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
NUM_ALGORITHME = 1

RNN_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 1
KEEP_PROBABILITY = 0.5
OPTIMIZER_TYPE = 'adam'
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 100
data_dir_path = '../input'
data_file = '/text-sumarization/articles3_token_stop.csv'
glove_file = '/glove-global-vectors-for-word-representation/glove.6B.' + str(EMBEDDING_SIZE) + 'd.txt'
VERBOSE = 1
NUM_LAYERS = 3
MAX_INPUT_SEQ_LENGTH = 500
MAX_TARGET_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_TARGET_VOCAB_SIZE = 2000
NUM_SAMPLES = 10000
MAX_DECODER_SEQ_LENGTH = 4
def def_keras_optimizer():
    if OPTIMIZER_TYPE == 'sgd':
        # default LEARNING_RATE = 0.01
        keras_optimizer = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.0, decay=0.0, nesterov=False)
    elif OPTIMIZER_TYPE == 'rmsprop':
        # default LEARNING_RATE = 0.001
        keras_optimizer = keras.optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=None, decay=0.0)
    elif OPTIMIZER_TYPE == 'adagrad':
        # default LEARNING_RATE = 0.01
        keras_optimizer = keras.optimizers.Adagrad(lr=LEARNING_RATE, epsilon=None, decay=0.0)
    elif OPTIMIZER_TYPE == 'adadelta':
        # default LEARNING_RATE = 1.0
        keras_optimizer = keras.optimizers.Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=None, decay=0.0)
    else:   # OPTIMIZER_TYPE == 'adam':
        # default LEARNING_RATE = 0.001
        keras_optimizer = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                                amsgrad=False)

    return keras_optimizer


def def_tf_optimizer(lr):
    if OPTIMIZER_TYPE == 'sgd':
        # default LEARNING_RATE = 0.01
        tf_optimizer = tf.train.GradientDescentOptimizer(lr)
    elif OPTIMIZER_TYPE == 'rmsprop':
        # default LEARNING_RATE = 0.001
        tf_optimizer = tf.train.RMSPropOptimizer(lr)
    elif OPTIMIZER_TYPE == 'adagrad':
        # default LEARNING_RATE = 0.01
        tf_optimizer = tf.train.AdagradOptimizer(lr)
    elif OPTIMIZER_TYPE == 'adadelta':
        # default LEARNING_RATE = 1.0
        tf_optimizer = tf.train.AdadeltaOptimizer(lr)
    else:   # OPTIMIZER_TYPE == 'adam':
        # default LEARNING_RATE = 0.001
        tf_optimizer = tf.train.AdamOptimizer(lr)

    return tf_optimizer
def preprocess_data_char(inputs, targets):
    input_texts = []
    target_texts = []
    characters = set()

    for line in inputs[:NUM_SAMPLES]:
        input_texts.append(line)
        for char in line:
            if char not in characters:
                characters.add(char)
    for line in targets[:NUM_SAMPLES]:
        line = '\t' + line + '\n'
        target_texts.append(line)
        for char in line:
            if char not in characters:
                characters.add(char)

    characters = sorted(list(characters))
    num_tokens = len(characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    char2idx = dict([(char, i) for i, char in enumerate(characters)])
    idx2char = dict((i, char) for char, i in char2idx.items())

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, char2idx[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, char2idx[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, char2idx[char]] = 1.
    config = dict()
    config['num_tokens'] = num_tokens
    config['encoder_input_data'] = encoder_input_data
    config['decoder_input_data'] = decoder_input_data
    config['decoder_target_data'] = decoder_target_data
    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    config['max_encoder_seq_length'] = max_encoder_seq_length
    config['max_decoder_seq_length'] = max_decoder_seq_length

    return config
class Seq2SeqChar(object):
    def __init__(self, config):
        self.num_tokens = config['num_tokens']
        self.encoder_input_data = config['encoder_input_data']
        self.decoder_input_data = config['decoder_input_data']
        self.decoder_target_data = config['decoder_target_data']
        self.char2idx = config['char2idx']
        self.idx2char = config['idx2char']
        self.max_encoder_seq_length = config['max_encoder_seq_length']
        self.max_decoder_seq_length = config['max_decoder_seq_length']

        encoder_inputs = Input(shape=(None, self.num_tokens))
        encoder = LSTM(RNN_SIZE, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None, self.num_tokens))
        decoder_lstm = LSTM(RNN_SIZE, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = def_keras_optimizer()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, validation_split=0.2)
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(RNN_SIZE,))
        decoder_state_input_c = Input(shape=(RNN_SIZE,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def summarize(self, input_seq):
        # Encode the input as state vectors.
        encoder_input = np.zeros((1, self.max_encoder_seq_length, self.num_tokens), dtype='float32')
        for i, char in enumerate(input_seq):
            encoder_input[0, i, self.char2idx[char]] = 1

        states_value = self.encoder_model.predict(encoder_input)

        target_seq = np.zeros((1, 1, self.num_tokens))
        target_seq[0, 0, self.char2idx['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
            sampled_char = self.idx2char[sampled_token_index]
            decoded_sentence += sampled_char

            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            target_seq = np.zeros((1, 1, self.num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]

        return decoded_sentence
def main_seq2seq_char():
    df = pd.read_csv(data_dir_path + data_file)
    targets = df['titre']
    inputs = df['article']
    config = preprocess_data_char(inputs, targets)
    summarize = Seq2SeqChar(config)
    for i in np.random.permutation(np.arange(len(inputs)))[0:20]:
        x = inputs[i]
        decoded_sentence = summarize.summarize(x)
        print('-')
        print('Input sentence:', x)
        print('Decoded sentence:', decoded_sentence)


def get_accuracy(target, logits):
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
def preprocess_data(input_texts, target_texts):
    vocab = set()
    vocab.add('<PAD>')
    vocab.add('END')
    vocab.add('<UNK>')
    vocab.add('START')

    for input_text in input_texts:
        for word in input_text:
            if word not in vocab:
                vocab.add(word)
    for target_text in target_texts:
        for word in target_text:
            if word not in vocab:
                vocab.add(word)

    vocab = sorted(list(vocab))
    word2idx = dict([(word, i) for i, word in enumerate(vocab)])
    idx2word = dict((i, word) for word, i in word2idx.items())

    source_text_id = []
    target_text_id = []

    for i in range(len(input_texts)):
        source_sentence = input_texts[i]
        target_sentence = target_texts[i]

        source_token_id = []
        target_token_id = []

        for index, token in enumerate(source_sentence):
            if token != "":
                source_token_id.append(word2idx[token])

        for index, token in enumerate(target_sentence):
            if token != "":
                target_token_id.append(word2idx[token])

        target_token_id.append(word2idx['END'])

        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)

    config = dict()
    config['word2idx'] = word2idx
    config['idx2word'] = idx2word
    config['source_text_id'] = source_text_id
    config['target_text_id'] = target_text_id

    return config
class TfSeq2Seq(object):

    def __init__(self, config):
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']
        self.source_text_id = config['source_text_id']
        self.target_text_id = config['target_text_id']

    def encoding_layer(self, input_data, keep_prob):
        embed = tf.contrib.layers.embed_sequence(input_data,
                                                 vocab_size=len(self.word2idx),
                                                 embed_dim=EMBEDDING_SIZE)

        stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(RNN_SIZE), keep_prob) for _ in
             range(NUM_LAYERS)])

        outputs, state = dynamic_rnn(stacked_cells, embed, dtype=tf.float32)
        return outputs, state

    def process_decoder_input(self, targets):
        go_id = self.word2idx['START']

        after_slice = tf.strided_slice(targets, [0, 0], [BATCH_SIZE, -1], [1, 1])
        after_concat = tf.concat([tf.fill([BATCH_SIZE, 1], go_id), after_slice], 1)

        return after_concat

    def decoding_layer(self, dec_input, encoder_state, keep_prob, target_sequence_length, max_target_len):
        target_vocab_size = len(self.word2idx)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, EMBEDDING_SIZE]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(RNN_SIZE) for _ in range(NUM_LAYERS)])

        with tf.variable_scope("decode"):
            output_layer = tf.layers.Dense(target_vocab_size)
            dec_cell_train = tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob=keep_prob)

            helper_train = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)

            decoder_train = tf.contrib.seq2seq.BasicDecoder(dec_cell_train, helper_train, encoder_state,
                                                            output_layer)

            train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished=True,
                                                                   maximum_iterations=max_target_len)

        with tf.variable_scope("decode", reuse=True):
            start_sequence_id = self.word2idx['START']
            end_sequence_id = self.word2idx['END']

            dec_cell_infer = tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob=keep_prob)

            helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([BATCH_SIZE],
                                                                                            start_sequence_id),
                                                                    end_sequence_id)

            decoder_infer = tf.contrib.seq2seq.BasicDecoder(dec_cell_infer, helper_infer, encoder_state, output_layer)

            infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
                                                                   maximum_iterations=max_target_len)

        return train_output, infer_output

    def seq2seq_model(self, input_data, targets, keep_prob, target_sequence_length, max_target_len):

        enc_outputs, enc_states = self.encoding_layer(input_data, keep_prob)

        dec_input = self.process_decoder_input(targets)

        train_output, infer_output = self.decoding_layer(dec_input, enc_states, keep_prob, target_sequence_length,
                                                         max_target_len)

        return train_output, infer_output

    def get_batches(self, sources, targets):
        for batch_i in range(0, len(sources) // BATCH_SIZE):
            start_i = batch_i * BATCH_SIZE

            sources_batch = sources[start_i:start_i + BATCH_SIZE]
            targets_batch = targets[start_i:start_i + BATCH_SIZE]

            pad_int = self.word2idx['<PAD>']
            max_sentence_source = max([len(sentence) for sentence in sources_batch])
            max_sentence_target = max([len(sentence) for sentence in targets_batch])
            pad_sources_batch = np.array([sentence + [pad_int] * (max_sentence_source - len(sentence)) for sentence in
                                          sources_batch])
            pad_targets_batch = np.array([sentence + [pad_int] * (max_sentence_target - len(sentence)) for sentence in
                                          targets_batch])

            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

    def fit(self):
        train_graph = tf.Graph()
        with train_graph.as_default():
            inputs = tf.placeholder(tf.int32, [None, None], name='input')
            input_data = tf.reverse(inputs, [-1])
            targets = tf.placeholder(tf.int32, [None, None], name='targets')
            target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
            max_target_len = tf.reduce_max(target_sequence_length)
            lr_rate = tf.placeholder(tf.float32, name='lr_rate')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            train_logits, inference_logits = self.seq2seq_model(input_data, targets, keep_prob, target_sequence_length,
                                                                max_target_len)
            training_logits = tf.identity(train_logits.rnn_output, name='logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
            masks = tf.sequence_mask(target_sequence_length, max_target_len, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):

                cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
                # optimizer = tf.train.AdamOptimizer(lr_rate)
                optimizer = def_tf_optimizer(lr_rate)
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not
                                    None]
                train_op = optimizer.apply_gradients(capped_gradients)

            train_source = self.source_text_id[BATCH_SIZE:]
            train_target = self.target_text_id[BATCH_SIZE:]
            valid_source = self.source_text_id[:BATCH_SIZE]
            valid_target = self.target_text_id[:BATCH_SIZE]

            (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = \
                next(self.get_batches(valid_source, valid_target))

            sess = tf.Session(graph=train_graph)
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(EPOCHS):
                for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                        self.get_batches(train_source, train_target)):
                    _, loss = sess.run([train_op, cost], {inputs: source_batch, targets: target_batch,
                                                          lr_rate: LEARNING_RATE,
                                                          target_sequence_length: targets_lengths,
                                                          keep_prob: KEEP_PROBABILITY})

                    if batch_i > 0:
                        if batch_i % 5 == 0:
                            batch_train_logits = sess.run(
                             inference_logits, {input_data: source_batch, target_sequence_length: targets_lengths,
                                                keep_prob: 1.0})

                            batch_valid_logits = sess.run(
                                inference_logits, {inputs: valid_sources_batch,
                                                   target_sequence_length:  valid_targets_lengths,
                                                   keep_prob: 1.0})

                            train_acc = get_accuracy(target_batch, batch_train_logits)
                            valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}'
                                  ', Loss: {:>6.4f}'.format(epoch_i, batch_i, len(self.source_text_id) // BATCH_SIZE,
                                                            train_acc, valid_acc, loss))
        return target_sequence_length, keep_prob, inputs, sess, inference_logits

    def summarize(self, inputs, text, sess, inference_logits, target_sequence_length, keep_prob):
        translate_sentence = []
        for word in text:
            if word in self.word2idx:
                translate_sentence.append(self.word2idx[word])
            else:
                translate_sentence.append(self.word2idx['<UNK>'])

        translate_logits = sess.run(inference_logits, {inputs: [translate_sentence] * BATCH_SIZE,
                                                       target_sequence_length: [len(
                                                           translate_sentence) * 2] * BATCH_SIZE,
                                                       keep_prob: 1.0})[0]

        return " ".join([self.idx2word[i] for i in translate_logits])
def main_tf_seq2seq():
    train = pd.read_csv(data_dir_path + data_file)
    resumes = []
    articles = []
    for resume in train[train.columns[0]].values:
        resumes.append(resume.split(' '))
    for article in train[train.columns[1]].values:
        articles.append(article.split(' '))

    config = preprocess_data(articles[:NUM_SAMPLES], resumes[:NUM_SAMPLES])
    summarizer = TfSeq2Seq(config)
    target_sequence_length, keep_prob, inputs, sess, inference_logits = summarizer.fit()
    for i in np.random.permutation(np.arange(len(articles)))[0:20]:
        x = articles[i]
        headline = summarizer.summarize(inputs, x, sess, inference_logits, target_sequence_length, keep_prob)
        print('Article: ', articles)
        print('Generated Headline: ', headline)
        print('Original Article: ', x)
def fit_text(x, y, input_seq_max_length=None, target_seq_max_length=None):
    if input_seq_max_length is None:
        input_seq_max_length = MAX_INPUT_SEQ_LENGTH
    if target_seq_max_length is None:
        target_seq_max_length = MAX_TARGET_SEQ_LENGTH
    input_counter = Counter()
    target_counter = Counter()
    max_input_seq_length = 0
    max_target_seq_length = 0

    for line in x:
        text = [word for word in line.split(' ')]
        for i, word in enumerate(text):
            if word == '':
                del text[i]
        seq_length = len(text)
        if seq_length > input_seq_max_length:
            text = text[0:input_seq_max_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_input_seq_length = max(max_input_seq_length, seq_length)

    for i, line in enumerate(y):

        line2 = 'START ' + str(line) + ' END'
        text = [word for word in line2.split(' ')]
        for j, word in enumerate(text):
            if word == '':
                del text[j]
        seq_length = len(text)
        if seq_length > target_seq_max_length:
            text = text[0:target_seq_max_length]
            seq_length = len(text)
        for word in text:
            target_counter[word] += 1
            max_target_seq_length = max(max_target_seq_length, seq_length)

    input_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):
        input_word2idx[word[0]] = idx + 2
    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])

    target_word2idx = dict()
    for idx, word in enumerate(target_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1
    target_word2idx['UNK'] = 0

    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    num_input_tokens = len(input_word2idx)
    num_target_tokens = len(target_word2idx)

    config = dict()
    config['input_word2idx'] = input_word2idx
    config['input_idx2word'] = input_idx2word
    config['target_word2idx'] = target_word2idx
    config['target_idx2word'] = target_idx2word
    config['num_input_tokens'] = num_input_tokens
    config['num_target_tokens'] = num_target_tokens
    config['max_input_seq_length'] = max_input_seq_length
    config['max_target_seq_length'] = max_target_seq_length

    return config
def transform_input_text(texts, input_word2idx, max_input_seq_length):
    temp = []
    for line in texts:
        x = []
        for word in line.lower().split(' '):
            wid = 1
            if word in input_word2idx:
                wid = input_word2idx[word]
            x.append(wid)
            if len(x) >= max_input_seq_length:
                break
        temp.append(x)
    temp = pad_sequences(temp, maxlen=max_input_seq_length)

    print(temp.shape)
    return temp
def transform_target_encoding(texts, max_target_seq_length):
    temp = []
    for line in texts:
        x = []
        line2 = 'START ' + line.lower() + ' END'
        for word in line2.split(' '):
            x.append(word)
            if len(x) >= max_target_seq_length:
                break
        temp.append(x)

    temp = np.array(temp)
    print(temp.shape)
    return temp
class RecursiveRNN(object):

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        if NUM_ALGORITHME == 3:
            inputs1 = Input(shape=(self.max_input_seq_length,))
            am1 = Embedding(self.num_input_tokens, 128)(inputs1)
            am2 = LSTM(128)(am1)

            inputs2 = Input(shape=(self.max_target_seq_length,))
            sm1 = Embedding(self.num_target_tokens, 128)(inputs2)
            sm2 = LSTM(128)(sm1)

            decoder1 = concatenate([am2, sm2])
            outputs = Dense(self.num_target_tokens, activation='softmax')(decoder1)
        elif NUM_ALGORITHME == 4:
            # article input model
            inputs1 = Input(shape=(self.max_input_seq_length,))
            article1 = Embedding(self.num_input_tokens, 128)(inputs1)
            article2 = Dropout(0.3)(article1)

            # summary input model
            inputs2 = Input(shape=(min(self.num_target_tokens, MAX_DECODER_SEQ_LENGTH),))
            summ1 = Embedding(self.num_target_tokens, 128)(inputs2)
            summ2 = Dropout(0.3)(summ1)
            summ3 = LSTM(128)(summ2)
            summ4 = RepeatVector(self.max_input_seq_length)(summ3)

            # decoder model
            decoder1 = concatenate([article2, summ4])
            decoder2 = LSTM(128)(decoder1)
            outputs = Dense(self.num_target_tokens, activation='softmax')(decoder2)
        else:
            # article input model
            inputs1 = Input(shape=(self.max_input_seq_length,))
            article1 = Embedding(self.num_input_tokens, 128)(inputs1)
            article2 = LSTM(128)(article1)
            article3 = RepeatVector(128)(article2)
            # summary input model
            inputs2 = Input(shape=(self.max_target_seq_length,))
            summ1 = Embedding(self.num_target_tokens, 128)(inputs2)
            summ2 = LSTM(128)(summ1)
            summ3 = RepeatVector(128)(summ2)
            # decoder model
            decoder1 = concatenate([article3, summ3])
            decoder2 = LSTM(128)(decoder1)
            outputs = Dense(self.num_target_tokens, activation='softmax')(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        optimizer = def_keras_optimizer()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def generate_batch(self, x_samples, y_samples, batch_size):
        encoder_input_data_batch = []
        decoder_input_data_batch = []
        decoder_target_data_batch = []
        line_idx = 0
        while True:
            for recordIdx in range(0, len(x_samples)):
                target_words = y_samples[recordIdx]
                x = x_samples[recordIdx]
                decoder_input_line = []

                for idx in range(0, len(target_words) - 1):
                    w2idx = 0  # default [UNK]
                    w = target_words[idx]
                    if w in self.target_word2idx:
                        w2idx = self.target_word2idx[w]
                    decoder_input_line = decoder_input_line + [w2idx]
                    decoder_target_label = np.zeros(self.num_target_tokens)
                    w2idx_next = 0
                    if target_words[idx + 1] in self.target_word2idx:
                        w2idx_next = self.target_word2idx[target_words[idx + 1]]
                    if w2idx_next != 0:
                        decoder_target_label[w2idx_next] = 1
                    decoder_input_data_batch.append(decoder_input_line)
                    encoder_input_data_batch.append(x)
                    decoder_target_data_batch.append(decoder_target_label)

                    line_idx += 1
                    if line_idx >= batch_size:
                        if NUM_ALGORITHME != 4:
                            yield [pad_sequences(encoder_input_data_batch, self.max_input_seq_length),
                                   pad_sequences(decoder_input_data_batch,
                                                 self.max_target_seq_length)], np.array(decoder_target_data_batch)
                        else:
                            yield [pad_sequences(encoder_input_data_batch, self.max_input_seq_length), pad_sequences(
                                decoder_input_data_batch, min(self.num_target_tokens, MAX_DECODER_SEQ_LENGTH))], \
                                  np.array(decoder_target_data_batch)
                        line_idx = 0
                        encoder_input_data_batch = []
                        decoder_input_data_batch = []
                        decoder_target_data_batch = []

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size):

        y_train = transform_target_encoding(y_train, self.max_target_seq_length)
        y_test = transform_target_encoding(y_test, self.max_target_seq_length)

        x_train = transform_input_text(x_train, self.input_word2idx, self.max_input_seq_length)
        x_test = transform_input_text(x_test, self.input_word2idx, self.max_input_seq_length)

        train_gen = self.generate_batch(x_train, y_train, batch_size)
        test_gen = self.generate_batch(x_test, y_test, batch_size)

        total_training_samples = sum([len(target_text) - 1 for target_text in y_train])
        total_testing_samples = sum([len(target_text) - 1 for target_text in y_test])
        train_num_batches = total_training_samples // batch_size
        test_num_batches = total_testing_samples // batch_size

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches, epochs=epochs, verbose=VERBOSE,
                                 validation_data=test_gen, validation_steps=test_num_batches)

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        start_token = self.target_word2idx['START']
        wid_list = [start_token]
        if NUM_ALGORITHME != 4:
            sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
        else:
            sum_input_seq = pad_sequences([wid_list], min(self.num_target_tokens, MAX_DECODER_SEQ_LENGTH))
        terminated = False

        target_text = ''

        while not terminated:
            output_tokens = self.model.predict([input_seq, sum_input_seq])
            sample_token_idx = np.argmax(output_tokens[0, :])
            sample_word = self.target_idx2word[sample_token_idx]
            wid_list = wid_list + [sample_token_idx]

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or len(wid_list) >= self.max_target_seq_length:
                terminated = True
            else:
                if NUM_ALGORITHME != 4:
                    sum_input_seq = pad_sequences([wid_list], self.max_target_seq_length)
                else:
                    sum_input_seq = pad_sequences([wid_list],  min(self.num_target_tokens, MAX_DECODER_SEQ_LENGTH))
        return target_text.strip()
def main_rnn():
    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + data_file)

    print('extract configuration from input texts ...')
    y = df['titre']
    x = df['article']
    config = fit_text(x, y)

    print('configuration extracted from input texts ...')

    summarizer = RecursiveRNN(config)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('demo size: ', len(x_train))
    print('testing size: ', len(x_test))

    print('start fitting ...')
    summarizer.fit(x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(x)))[0:20]:
        x = x[i]
        actual_headline = y[i]
        headline = summarizer.summarize(x)
        # print('Article: ', x)
        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)
def load_glove():
    with open(data_dir_path + glove_file, 'r') as f:
        word2vector = {}
        for line in f:
            line_ = line.strip()    # Remove white space
            words_vec = line_.split()
            word2vector[words_vec[0]] = np.array(words_vec[1:], dtype=float)
    return word2vector
def transform_input_text_glove(texts, max_input_seq_length, unknown_emb, word2em):
    temp = []
    for line in texts:
        x = np.zeros(shape=(max_input_seq_length, EMBEDDING_SIZE))
        for idx, word in enumerate(line.lower().split(' ')):
            if idx >= max_input_seq_length:
                break
            emb = unknown_emb
            if word in word2em:
                emb = word2em[word]
            x[idx, :] = emb
        temp.append(x)
    temp = pad_sequences(temp, maxlen=max_input_seq_length)

    print(temp.shape)
    return temp
class Seq2SeqSummarizer(object):

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        if NUM_ALGORITHME == 6:
            encoder_inputs = Input(shape=(None,), name='encoder_inputs')
            encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=RNN_SIZE,
                                          input_length=self.max_input_seq_length, name='encoder_embedding')
            encoder_lstm = LSTM(units=RNN_SIZE, return_state=True, name='encoder_lstm')
            encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
            encoder_states = [encoder_state_h, encoder_state_c]

            decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
            decoder_lstm = LSTM(units=RNN_SIZE, return_state=True, return_sequences=True, name='decoder_lstm')
            decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)
            decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)
        else:
            encoder_inputs = Input(shape=(None, EMBEDDING_SIZE), name='encoder_inputs')
            encoder_lstm = LSTM(units=RNN_SIZE, return_state=True, name='encoder_lstm')
            encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
            encoder_states = [encoder_state_h, encoder_state_c]

            decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
            decoder_lstm = LSTM(units=RNN_SIZE, return_state=True, return_sequences=True, name='decoder_lstm')
            decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                             initial_state=encoder_states)
            decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
            decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = def_keras_optimizer()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(RNN_SIZE,)), Input(shape=(RNN_SIZE,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_glove(self):
        self.word2em = load_glove()

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length,
                                                            self.num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length,
                                                           self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size):

        y_train = transform_target_encoding(y_train, self.max_target_seq_length)
        y_test = transform_target_encoding(y_test, self.max_target_seq_length)

        if NUM_ALGORITHME == 6:
            x_train = transform_input_text(x_train, self.input_word2idx, self.max_input_seq_length)
            x_test = transform_input_text(x_test, self.input_word2idx, self.max_input_seq_length)
        else:
            x_train = transform_input_text_glove(x_train, self.max_input_seq_length, self.unknown_emb, self.word2em)
            x_test = transform_input_text_glove(x_test, self.max_input_seq_length, self.unknown_emb, self.word2em)

        train_gen = self.generate_batch(x_train, y_train, batch_size)
        test_gen = self.generate_batch(x_test, y_test, batch_size)

        train_num_batches = len(x_train) // batch_size
        test_num_batches = len(x_test) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()

    def summarize_glove(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()
def main_seq2seq():

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + data_file)

    print('extract configuration from input texts ...')
    y = df['titre']
    x = df['article']
    config = fit_text(x, y)

    print('configuration extracted from input texts ...')

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_glove()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('training size: ', len(x_train))
    print('testing size: ', len(x_test))

    print('start fitting ...')
    summarizer.fit(x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

    for i in np.random.permutation(np.arange(len(x)))[0:20]:
        _x = x[i]
        actual_headline = y[i]
        if NUM_ALGORITHME == 6:
            headline = summarizer.summarize(_x)
        else:
            headline = summarizer.summarize_glove(_x)
        print('Generated Headline: ', headline)
        print('Original Headline: ', actual_headline)
if NUM_ALGORITHME == 1:
    main_seq2seq_char()
elif NUM_ALGORITHME == 2:
    main_tf_seq2seq()
elif 2 < NUM_ALGORITHME < 6:
    main_rnn()
else:
    main_seq2seq()
    