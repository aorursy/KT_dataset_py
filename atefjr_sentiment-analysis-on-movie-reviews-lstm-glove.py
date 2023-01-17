import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import gensim
from nltk.stem import WordNetLemmatizer
pd.options.display.width = None
# Load data
data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',sep='\t')
# View first 10 rows
print(data.head(10))
# Info about training data
print(data.describe())
print(data.info())
print(data.isnull().sum())
# Class counts
sns.countplot(data['Sentiment']).set_title("Count plot for sentiment classes")
plt.show()
data['text_length'] = data['Phrase'].apply(lambda x: len(x.split()))
# Remove empty phrase rows
print(data.loc[data['text_length'] == 0])
data = data[data['text_length'] != 0]
print(data.describe())
print(data[['text_length','Phrase']].head(10))
print(data['text_length'].describe())
# Text length histogram
sns.countplot(data['text_length']).set_title("Count plot for text length")
plt.show()
# Pre-Processing
X = []
for row in data['Phrase']:
    #row = re.sub('[^a-zA-Z]', ' ',row)
    row = row.lower()
    row = row.split()
    lemm = WordNetLemmatizer()
    row = [lemm.lemmatize(w) for w in row]
    X.append(' '.join(row))

data['Process_Phrase'] = X
data = data.drop_duplicates(subset=['Process_Phrase', 'Sentiment'])
print(data.head(10))
print(data.describe())
X = data['Process_Phrase']
sen = data['Sentiment']
nb_classes = 5
#One-hot vectors
Y = np.eye(nb_classes)[sen]

print(X[0:10])
print()
print(Y[0:10])
import tensorflow as tf
# Configurations
tf.app.flags.DEFINE_string("rnn_unit", 'lstm', "Type of RNN unit: rnn|gru|lstm.")
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm (clipping ratio).")
tf.app.flags.DEFINE_integer("num_epochs", 6, "Number of epochs during training.")
tf.app.flags.DEFINE_integer("batch_size", 264, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_hidden_units", 300, "Number of hidden units in each RNN cell (i/p vector length).")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_float("dropout", 0.4, "Amount to drop during training.")
tf.app.flags.DEFINE_integer("num_classes", 5, "Number of classification classes.")
tf.app.flags.DEFINE_string('f', '', 'kernel')
CONFIGS = tf.app.flags.FLAGS
def load_data_and_labels(X,Y):
    # Load the data
    X = [s.strip() for s in X]
    X = [s.replace("\"", "") for s in X]
    X = [[w for w in sent.strip().split()] for sent in X]
    #Y = [w.replace("\n", '') for w in Y]
    #Y = [[float(w) for w in sent.split()] for sent in Y]
    return X, Y


def load_glove_model():
    glove_model = gensim.models.KeyedVectors.load_word2vec_format("../input/stanfords-glove-pretrained-word-vectors/glove.6B.300d.txt")
    print("GLOVE MODEL LOADED")
    return glove_model


def sentence_to_vectors(sentence, glove_model, num_hidden_units):
    return [glove_model.wv[word].tolist() if word in glove_model.wv.vocab else [0.0] * num_hidden_units for word in sentence]


def data_to_vectors(X, glove_model, num_hidden_units):
    max_len = max(len(sentence) for sentence in X)

    data_as_vectors = []
    for line in X:
        vectors = sentence_to_vectors(line, glove_model, num_hidden_units)
        # Padding
        data_as_vectors.append(vectors + [[0.0] * num_hidden_units] * (max_len - len(line)))

    return data_as_vectors


def data_to_seqs(X):

    seq_lens = []

    for line in X:
        seq_lens.append(len(line))

    return seq_lens

def generate_epoch(X, y, seq_lens, num_epochs, batch_size):
    for epoch_num in range(num_epochs):
        yield generate_batch(X, y, seq_lens, batch_size)


def generate_batch(X, y, seq_lens, batch_size):
    data_size = len(X)

    num_batches = (data_size // batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield X[start_index:end_index], y[start_index:end_index], seq_lens[start_index:end_index]

def rnn_cell(CONFIGS, dropout):
    # Choosing cell type
    # Default activation is Tanh
    if CONFIGS.rnn_unit == 'rnn':
        rnn_cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif CONFIGS.rnn_unit == 'gru':
        rnn_cell_type = tf.nn.rnn_cell.GRUCell
    elif CONFIGS.rnn_unit == 'lstm':
        rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("Choose a valid RNN cell type.")

    # Create a single cell
    single_cell = rnn_cell_type(CONFIGS.num_hidden_units)

    # Apply dropoutwrapper to RNN cell (Only output dropout is applied)
    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=1 - dropout)

    # Stack cells on each other (Layers)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in range(CONFIGS.num_layers)])

    return stacked_cell


# Softmax layer
def rnn_softmax(CONFIGS, outputs):
    # Variable scopes is a way to share variable among different parts of the code
    # helps in initializing variables in one place and reuse them in different parts of code

    with tf.variable_scope('rnn_softmax', reuse=True):
        W_softmax = tf.get_variable("W_softmax", [CONFIGS.num_hidden_units, CONFIGS.num_classes])
        b_softmax = tf.get_variable("b_softmax", [CONFIGS.num_classes])

    logits = tf.matmul(outputs, W_softmax) + b_softmax

    return logits


class model(object):

    def __init__(self, CONFIGS):

        # Placeholders
        self.inputs_X = tf.placeholder(tf.float32, shape=[None, None, CONFIGS.num_hidden_units], name='inputs_X')
        self.targets_y = tf.placeholder(tf.float32, shape=[None, None], name='targets_y')
        self.seq_lens = tf.placeholder(tf.int32, shape=[None], name='seq_lens')
        self.dropout = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        with tf.name_scope("rnn"):
            # Create folded RNN network (depth) [RNN cell * num_layers]
            stacked_cell = rnn_cell(CONFIGS, self.dropout)


            # Initial state is zero for each i/p batch as each input example is independent on the other
            initial_state = stacked_cell.zero_state(self.batch_size, tf.float32)

        # Unfold RNN cells in time axis

        # sequence_length ->  An int32/int64 vector sized [batch_size]
        # is used to copy-through state and zero-out outputs
        # when past a batch element's sequence length.

        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'state' is a tuple of shape [num_layers, batch_size, cell_state_size]
        #  state[0] is the state from first RNN layer, state[-1] is the state from last RNN layer

            all_outputs, state = tf.nn.dynamic_rnn(cell=stacked_cell, inputs=self.inputs_X, initial_state=initial_state,
                                                sequence_length=self.seq_lens, dtype=tf.float32)

        # Since we are using variable length inputs padded to maximum input length and we are feeding
        # sequence_length to tf.nn.dynamic_rnn, Outputs after input seq. length will be 0, and last state will
        # be propagated, So we can't use output[:,-1,:], instead we will use state[-1]
            if CONFIGS.rnn_unit == 'lstm':
                outputs = state[-1][1]
            else:
                outputs = state[-1]
        # Process RNN outputs
        with tf.variable_scope('rnn_softmax'):
            W_softmax = tf.get_variable("W_softmax", [CONFIGS.num_hidden_units, CONFIGS.num_classes])
            b_softmax = tf.get_variable("b_softmax", [CONFIGS.num_classes])

        # Softmax layer
        # logits [batch_size, num_classes]
        with tf.name_scope("logits"):
            logits = rnn_softmax(CONFIGS, outputs)
            # Convert logits into probabilities
            self.probabilities = tf.nn.softmax(logits)

        with tf.name_scope("accuracy"):
            # Array of boolean
            correct_prediction = tf.equal(tf.argmax(self.targets_y, 1), tf.argmax(self.probabilities, 1))
            # Number of correct examples / batch size
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # Loss
        with tf.name_scope("loss"):
            # Multi-class - One label - Mutually exclusive classification, so we use
            # softmax cross entropy cost function
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.targets_y))

        ####################################################################


        # Optimization
        with tf.name_scope("optimizer"):
            # Define learning rate (Updated each epoch)
            self.lr = tf.Variable(0.0, trainable=False)
            trainable_vars = tf.trainable_variables()

            # clip the gradient to avoid vanishing or blowing up gradients
            # max_gradient_norm/sqrt(add each element square))
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), CONFIGS.max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))

        ####################################################################
        # for model saving
        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(tf.global_variables())

    def step(self, sess, batch_X, batch_seq_lens, batch_y=None, dropout=0.0, forward_only=True, predict=False, batch_size=1.0):

        input_feed = {self.inputs_X: batch_X, self.targets_y: batch_y, self.seq_lens: batch_seq_lens,
                      self.dropout: dropout, self.batch_size: batch_size}

        if forward_only:
            if not predict:
                output_feed = [self.accuracy]
            elif predict:
                input_feed = {self.inputs_X: batch_X, self.seq_lens: batch_seq_lens,
                              self.dropout: dropout, self.batch_size: batch_size}
                output_feed = [self.probabilities]
        else:  # training
            output_feed = [self.train_optimizer, self.loss, self.accuracy]

        outputs = sess.run(output_feed, input_feed)

        if forward_only:
            return outputs[0]
        else:  # training
            return outputs[0], outputs[1], outputs[2]
def predict(tf_model, sess, glove_model, input):
    sentence = [w for w in input.strip().split()]
    data = [glove_model.wv[word].tolist() if word in glove_model.wv.vocab else [0.0] * CONFIGS.num_hidden_units for word
            in sentence]
    seqs = len(sentence)
    data = np.array(data)
    data = data[np.newaxis, :, :]
    seqs = np.array(seqs)
    seqs = seqs[np.newaxis]
    probabilities = tf_model.step(sess, batch_X=data,
                               batch_seq_lens=seqs,
                               forward_only=True, predict=True)
    predict_class = np.asscalar(np.argmax(probabilities, 1))
    return predict_class


def create_model(sess, CONFIGS):
    text_model = model(CONFIGS)
    print("Created new model.")
    sess.run(tf.global_variables_initializer())

    return text_model


def run_model(X,Y):
    tf.reset_default_graph()
    train_X, train_y = load_data_and_labels(X,Y)
    glove_model = load_glove_model()
    train_seq_lens = data_to_seqs(train_X)
    print("DATA IS LOADED")
    with tf.Session() as sess:
        # Load old model or create new one
        model = create_model(sess, CONFIGS)

        print("STARTING TRAINING")
        print("----------------------")

        # Train results
        for epoch_num, epoch in enumerate(generate_epoch(train_X, train_y, train_seq_lens,
                                                         CONFIGS.num_epochs, CONFIGS.batch_size)):
            print("EPOCH #%i started:" % (epoch_num + 1))
            print("----------------------")

            # Assign learning rate
            sess.run(tf.assign(model.lr, CONFIGS.learning_rate *
                               (CONFIGS.learning_rate_decay_factor ** epoch_num)))

            train_loss = []
            train_accuracy = []
            curr_time = dt.datetime.now()
            for batch_num, (batch_X, batch_y, batch_seq_lens) in enumerate(epoch):
                data = data_to_vectors(batch_X, glove_model, CONFIGS.num_hidden_units)
                _, loss, accuracy = model.step(sess, data, batch_seq_lens, batch_y, dropout=CONFIGS.dropout,
                                                     forward_only=False, batch_size=CONFIGS.batch_size)

                train_loss.append(loss)
                train_accuracy.append(accuracy)
                #print("Epoch {}, Step {}, loss: {:.3f}, accuracy: {:.3f}".format(epoch_num+1,batch_num, loss, accuracy))



            seconds = (float((dt.datetime.now() - curr_time).seconds))
            print()
            print("EPOCH #%i SUMMARY" % (epoch_num + 1))
            print("Total Average Training loss %.3f" % np.mean(train_loss))
            print("Total Average Training accuracy %.3f" % np.mean(train_accuracy))
            print("Time taken (seconds) %.3f" % seconds)
            print("----------------------")
        print("TRAINING ENDED")
        print("----------------------")

        #################################################
        data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
        # Test data Pre-Processing
        predicted = []
        for row in data['Phrase']:
            # row = re.sub('[^a-zA-Z]', ' ',row)
            row = row.lower()
            row = row.split()
            lemm = WordNetLemmatizer()
            row = [lemm.lemmatize(w) for w in row]
            if len(row) == 0:
                predicted.append(2)
            else:
                predicted.append(predict(model, sess, glove_model, ' '.join(row)))

        data['Sentiment'] = predicted

        data.drop(['Phrase', 'SentenceId'], axis=1, inplace=True)
        print(data.head(10))
        data.to_csv('Submission.csv', header=True, index=None, sep=',')
run_model(X,Y)