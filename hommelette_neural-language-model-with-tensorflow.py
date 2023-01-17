!pip install tensorflow==1.15.2
import collections
import functools
import nltk
import re
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
tf.__version__
nltk.download('punkt')
df = pd.read_csv('../input/better-donald-trump-tweets/Donald-Tweets!.csv')
text = df.Tweet_Text.tolist()
text[:5]
def sent_to_tokens(sent, tokenizer):
    words = tokenizer.tokenize(sent)
    tokens = []
    for w in words:
        w = re.sub(r'https?://\S+', '<URL>', w)
        w = re.sub(r'#\S+', '<TOPIC>', w)
        w = re.sub(r'@\S+', '<USER>', w)
        tokens.append(w)
    return tokens


def basic_text_preprocess(text):
    custom_tokenize = functools.partial(sent_to_tokens, tokenizer=nltk.TweetTokenizer())
    tokens = [custom_tokenize(s) for s in text]
    print(f'num of docs: {len(tokens)}')
    return tokens

tokens = basic_text_preprocess(text)
print('before normalizaion:')
print(text[0])
print('after normalization:')
print(tokens[0])
def build_vocab(tokens, top_pct=None):
    tok2idx = collections.defaultdict(lambda: 0)
    unique_tokens = list(functools.reduce(lambda a, b: set(a).union(b), tokens))
    if top_pct is not None:
        counts = collections.Counter(functools.reduce(lambda a, b: a + b, tokens))
        max_len = int(len(unique_tokens) * top_pct)
        unique_tokens = [*map(lambda wc: wc[0], counts.most_common(max_len))]
    print(f'vocab size: {len(unique_tokens)}')
    idx2tok = ['<UNK>', '<START>', '<END>', '<PAD>'] + unique_tokens
    for i, tok in enumerate(idx2tok):
        tok2idx[tok] = i
    return tok2idx, idx2tok

tok2idx, idx2tok = build_vocab(tokens, 0.9)
def batches_generator(batch_size, tokens, tok2idx):
    n_samples = len(tokens)
    order = np.random.permutation(n_samples) # shuffle data
    n_batches = n_samples // batch_size + 1
    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list, y_list, max_len = [], [], 0
        for idx in order[batch_start:batch_end]:
            idx_lst = [tok2idx[t] for t in tokens[idx]]
            x = [tok2idx['<START>']] + idx_lst
            x_list.append(x)
            y = idx_lst + [tok2idx['<END>']]
            y_list.append(y)
            max_len = max(max_len, len(x))
        X = np.ones([current_batch_size, max_len], dtype=np.int32) * tok2idx['<PAD>']
        Y = np.ones([current_batch_size, max_len], dtype=np.int32) * tok2idx['<PAD>']
        actual_lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            tok_len = len(x_list[n])
            actual_lengths[n] = tok_len
            X[n, :tok_len] = x_list[n]
            Y[n, :tok_len] = y_list[n]
        yield X, Y, actual_lengths
X, Y, actual_lengths = next(batches_generator(32, tokens, tok2idx))
print(X.shape)
print(Y.shape)
print(actual_lengths)
class NeuralLanguageModel:
    pass
def declare_placeholders(self):
    self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
    self.target_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_batch')
    self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')
    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[], name='dropout_rate')
    self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

NeuralLanguageModel.__declare_placeholders = classmethod(declare_placeholders)
def build_layers(self, vocabulary_size, embedding_dim, n_hidden, n_layers=2, cell_type='rnn'):
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    embedding_matrix = tf.Variable(initial_embedding_matrix, dtype=tf.float32, name='embedding_matrix')
    
    # two options are defined here, one is the vanilla rnn, another is lstm. Yet feel free to explore GRU etc.
    if cell_type == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_hidden, activation=tf.nn.tanh)
    elif cell_type == 'lstm':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    else:
        raise ValueError('Undefined rnn cell type')
    
    # apply dropout regularization, recall we defined placeholder for feeding dropout prob
    regulrized_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=rnn_cell, input_keep_prob=self.dropout_ph, output_keep_prob=self.dropout_ph,
        state_keep_prob=self.dropout_ph)
    
    mul_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[regulrized_rnn_cell]*n_layers)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, self.input_batch)
    output, state = tf.nn.dynamic_rnn(cell=mul_cell,
                                      inputs=embeddings,
                                      sequence_length=self.lengths,
                                      dtype=tf.float32)
    self.logits = tf.layers.dense(output, vocabulary_size, activation=None)

NeuralLanguageModel.__build_layers = classmethod(build_layers)
def compute_predictions(self):
    softmax_output = tf.nn.softmax(self.logits)
    self.predictions = tf.argmax(softmax_output, axis=-1)
    self.probs = softmax_output


def compute_loss(self, vocabulary_size, pad_index):
    # with tf.device('/device:GPU:0'):
    targets_one_hot = tf.one_hot(self.target_batch, vocabulary_size)
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets_one_hot,
                                                            logits=self.logits)
    mask = tf.cast(tf.not_equal(self.input_batch, pad_index), tf.float32)
    self.loss = tf.reduce_mean(tf.boolean_mask(loss_tensor, mask))

NeuralLanguageModel.__compute_predictions = classmethod(compute_predictions)
NeuralLanguageModel.__compute_loss = classmethod(compute_loss)
def perform_optimization(self):
    # with tf.device('/device:GPU:0'):
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
    grads_and_vars = self.optimizer.compute_gradients(self.loss)

    clip_norm = tf.cast(5.0, tf.float32) # clip gradient norm at 5.
    self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads_and_vars]

    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

NeuralLanguageModel.__perform_optimization = classmethod(perform_optimization)
def train_on_batch(self, sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_prob):
    feed_dict = {
        self.input_batch: x_batch,
        self.target_batch: y_batch,
        self.learning_rate_ph: learning_rate,
        self.dropout_ph: dropout_keep_prob,
        self.lengths: lengths
    }
    sess.run(self.train_op, feed_dict=feed_dict)
    return sess.run(self.loss, feed_dict=feed_dict)

NeuralLanguageModel.train_on_batch = classmethod(train_on_batch)
def init_model(self, vocabulary_size, embedding_dim, n_hidden, pad_index, n_layers, cell_type):
    self.__declare_placeholders()
    self.__build_layers(vocabulary_size, embedding_dim, n_hidden, n_layers, cell_type)
    self.__compute_predictions()
    self.__compute_loss(vocabulary_size, pad_index)
    self.__perform_optimization()

NeuralLanguageModel.__init__ = classmethod(init_model)
tf.reset_default_graph()

# network init parameters
params = {
    'vocabulary_size': len(tok2idx),
    'embedding_dim': 256,
    'n_hidden': 256,
    'n_layers': 2,
    'cell_type': 'rnn',
    'pad_index': tok2idx['<PAD>']
}

model = NeuralLanguageModel(**params)

# training parameters
batch_size = 32
n_epochs = 20
learning_rate = 0.005
learning_rate_decay = 1.03
dropout_keep_probability = 0.9
early_stopping = None

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_losses = []
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    
    # Train the model
    losses = []
    for x_batch, y_batch, lengths in batches_generator(batch_size, tokens, tok2idx):
        batch_loss = model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate,
                                          dropout_keep_probability)
        losses.append(batch_loss * len(x_batch))
    epoch_loss = np.sum(losses)/len(tokens)
    print(f'traning loss: {epoch_loss}')
    training_losses.append(epoch_loss)
    if early_stopping and epoch_loss == np.max(training_losses[-early_stopping:]):
        break
        
    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay
    
print('...training finished.')
input_seq = tokens[0]
x = [[tok2idx['<START>']] + [*map(tok2idx.get, input_seq)]]
feed_dict = {model.input_batch: x, model.lengths: [*map(len, x)]}
probs = sess.run(model.probs, feed_dict=feed_dict)
preds = [idx2tok[i] for i in probs[0].argmax(axis=1)]
keys = ['prediction', 'true token']
vals = [preds[:-1], input_seq]
pd.DataFrame(dict(list(zip(keys, vals))))
def get_cond_prob(sess, model, seq_toks):
    feed_dict = {
        model.input_batch: seq_toks,
        model.lengths: [len(seq_toks)+10] * len(seq_toks)
    }
    return np.log1p(sess.run(model.probs, feed_dict=feed_dict)[:, -1])


def find_topk_2d(arr, topk):
    order = arr.reshape(-1,).argsort()[::-1][:topk]
    idx = np.unravel_index(order, arr.shape)
    vals = []
    for i in range(topk):
        vals.append(arr[idx[0][i], idx[1][i]])
    return vals, idx


def init_beam_search(sess, model, start_toks, topk):
    cond_probs = get_cond_prob(sess, model, start_toks)
    vals, idx = find_topk_2d(cond_probs, topk)
    seq_toks = np.column_stack((np.repeat(start_toks, topk, axis=0),
                                idx[1].reshape(-1,1)))
    seq_probs = np.reshape(vals, (-1, 1))
    return seq_toks, seq_probs


def extend_seq(seq_toks, idx):
    new_seq = []
    for i in range(len(seq_toks)):
        seq_idx = idx[0][i]
        tok_idx = idx[1][i]
        new_seq.append(np.array(list(seq_toks[seq_idx]) + [tok_idx]))
    return np.array(new_seq)


def calculate_seq_probs(seq_probs, cond_probs):
    return cond_probs + seq_probs.reshape(-1, 1)
    

def itrate_search(sess, seq_toks, seq_probs, topk):
    cond_probs = get_cond_prob(sess, model, seq_toks)
    seq_probs = calculate_seq_probs(seq_probs, cond_probs)
    vals, idx = find_topk_2d(seq_probs, topk)
    seq_toks = extend_seq(seq_toks, idx)
    seq_probs = seq_probs[idx[0], idx[1]]
    return seq_toks, seq_probs

def beam_search(sess, model, tok2idx, idx2tok, start_str, beam_width):
    start_toks = [[tok2idx['<START>']] + [tok2idx[w] for w in start_str.split()]]
    detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
    seq_toks, seq_probs = init_beam_search(sess, model, start_toks, beam_width)
    for i in range(10):
        seq_toks, seq_probs = itrate_search(sess, seq_toks, seq_probs, beam_width)
    output_sents = []
    for seq in seq_toks:
        seq = seq[1:list(seq).index(tok2idx['<END>'])] # end seq at <END>
        sent = detokenizer.detokenize([idx2tok[i] for i in seq])
        output_sents.append(sent)
    return output_sents, seq_toks, seq_probs
# give some starting words
start_str = 'Love'
beam_width = 3
output_sents, seq_toks, seq_probs = beam_search(sess, model, tok2idx, idx2tok, start_str, beam_width)
# the most likely 3 output sentences
output_sents