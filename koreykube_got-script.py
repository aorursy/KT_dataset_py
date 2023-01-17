from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
b_size = 16
s_size = 32
training_file = "/kaggle/input/game-of-thrones-srt/season1.json"
def get_data_from_training_file(filename, batch_size, sequence_size):
    lines_array = []
    words_array = []
    df = pd.read_json(filename)
    df = df.dropna()
    episodes = list(df)

    for episode in episodes:
        for line in range(len(df[episode].values) - 1):
            lines_array.append(df[episode].values[line])

    for i in range(len(lines_array)):
        line = lines_array[i]
        words = line.split()
        for word in words:
            words_array.append(word)
    
    word_count = Counter(words_array)
    sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {k:w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w:k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    
    int_text = [vocab_to_int[w] for w in words_array]
    num_batches = int(len(int_text) / (sequence_size * batch_size))
    input_text = int_text[:num_batches * batch_size * sequence_size]
    
    output_text = np.zeros_like(input_text)
    output_text[:-1] = input_text[1:]
    output_text[-1] = input_text[0]
    input_text = np.reshape(input_text, (batch_size, -1))
    output_text = np.reshape(output_text, (batch_size, -1))
    
    return int_to_vocab, vocab_to_int, n_vocab, input_text, output_text
    
def get_batches(input_text, output_text, batch_size, sequence_size):
    num_batches = np.prod(input_text.shape) // (sequence_size * batch_size)
    for i in range(0, num_batches * sequence_size, sequence_size):
        yield input_text[:, i:i+sequence_size], output_text[:, i:i+sequence_size]
embedding_size = 128
lstm_size = 128
dropout_keep_prob = 0.7
def network(batch_size, sequence_size, embedding_size, lstm_size, keep_prob, n_vocab, reuse=False):
    with tf.compat.v1.variable_scope('LSTM', reuse=reuse):
        in_op = tf.compat.v1.placeholder(tf.int32, [None, sequence_size])
        out_op = tf.compat.v1.placeholder(tf.int32, [None, sequence_size])
        embedding = tf.compat.v1.get_variable('embedding_weights', [n_vocab, embedding_size])
        embed = tf.compat.v1.nn.embedding_lookup(embedding, in_op)
        lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_size)
        initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
        output, state = tf.compat.v1.nn.dynamic_rnn(lstm, embed, initial_state=initial_state, dtype=tf.float32)
        logits = tf.compat.v1.layers.dense(output, n_vocab, reuse=reuse)
        preds = tf.compat.v1.nn.softmax(logits)
        
        return in_op, out_op, lstm, initial_state, state, preds, logits
gradients_norm = 5
def get_loss_and_training_optimizer(out_op, logits, gradients_norm):
    loss_op = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=out_op, logits=logits))
    trainable_vars = tf.compat.v1.trainable_variables()
    grads, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(loss_op, trainable_vars), gradients_norm)
    opt = tf.compat.v1.train.AdamOptimizer()
    train_op = opt.apply_gradients(zip(grads, trainable_vars))
    
    return loss_op, train_op
num_epochs = 200
initial_words = ['I', 'will']
predict_top_k = 5
def main(loadCheckpoint):
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_training_file(training_file, b_size, s_size)
    in_op, out_op, lstm, initial_state, state, preds, logits = network(b_size, s_size, embedding_size, lstm_size, dropout_keep_prob, n_vocab)
    val_in_op, _, _, val_initial_state, val_state, val_preds, _ = network(1, 1, embedding_size, lstm_size, dropout_keep_prob, n_vocab, reuse=True)
    loss_op, train_op = get_loss_and_training_optimizer(out_op, logits, gradients_norm)
    
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    
    if not os.path.exists('training_checkpoints/'):
        os.mkdir('training_checkpoints/')
    
    if loadCheckpoint:
        checkpoint = tf.train.Checkpoint(optimizer = train_op, model = lstm)
        status = checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints/'))
        status.initialize_or_restore(sess)
    sess.run(tf.compat.v1.global_variables_initializer())
    iteration = 0
    
    for e in range(num_epochs):
        batches = get_batches(in_text, out_text, b_size, s_size)
        new_state = sess.run(initial_state)
        for x, y in batches:
            iteration += 1
            loss, new_state, _ = sess.run(
            [loss_op, state, train_op],
            feed_dict={in_op: x, out_op: y, initial_state: new_state})
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, num_epochs),
                     'Iteration: {}'.format(iteration),
                     'Loss: {:.4f}'.format(loss))
            if iteration % 1000 == 0:
                predict(initial_words, predict_top_k, sess, val_in_op, val_initial_state, val_preds, val_state, n_vocab, vocab_to_int, int_to_vocab)
                saver.save(sess, os.path.join('training_checkpoints/', 'model-{}.ckpt'.format(iteration)))
def predict(initial_words, predict_top_k, sess, in_op,
            initial_state, preds, state, n_vocab, vocab_to_int, int_to_vocab):
  new_state = sess.run(initial_state)
  words = initial_words
  samples = [w for w in words]
  for word in words:
    x = np.zeros((1, 1))
    x[0, 0] = vocab_to_int[word]
    pred, new_state = sess.run([preds, state], feed_dict={in_op: x, initial_state: new_state})

  def get_word(pred):
    p = np.squeeze(pred)
    p[p.argsort()][:-predict_top_k] = 0
    p = p / np.sum(p)
    word = np.random.choice(n_vocab, 1, p=p)[0]
    return word

  word = get_word(pred)

  n_samples = 200
  samples.append(int_to_vocab[word])
  for _ in range(n_samples):
    x[0, 0] = word
    pred, new_state = sess.run([preds, state], feed_dict={in_op: x, initial_state: new_state})
    word = get_word(pred)
    samples.append(int_to_vocab[word])

  print(' '.join(samples))
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
main(False)