# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import os

import shutil

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def download_and_read(dataset_dir, num_pairs=None):

    sent_filename = os.path.join(dataset_dir, "treebank-sents.txt")

    poss_filename = os.path.join(dataset_dir, "treebank-poss.txt")

    

    if not(os.path.exists(sent_filename) and os.path.exists(poss_filename)):

        if not os.path.exists(dataset_dir):

            os.makedirs(dataset_dir)

        fsents = open(sent_filename, "w")

        fposs = open(poss_filename, "w")

        sentences = nltk.corpus.treebank.tagged_sents()

        for sen in sentences:

            fsents.write(" ".join([w for w,p in sen])+"\n")

            fposs.write(" ".join([p for w,p in sen])+"\n")

        fsents.close()

        fposs.close()

        

    sents, poss = [], []

    with open(sent_filename, "r") as fsent:

        for idx, line in enumerate(fsent):

            sents.append(line.strip())

            if num_pairs is not None and idx >= num_pairs:

                break

    with open(poss_filename, "r") as fposs:

        for idx, line in enumerate(fposs):

            poss.append(line.strip())

            if num_pairs is not None and idx >= num_pairs:

                break

    return sents, poss

        

            
def preprocess_input(texts, vocab_size=None, lower=True):

    if vocab_size is None:

        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=lower)

    else:

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size+1,

                                                         oov_token='UNK', lower=lower)

    tokenizer.fit_on_texts(texts)

    if vocab_size is not None:

        tokenizer.word_index = {e:i for e,i in 

                               tokenizer.word_index.items() if 

                               i <= vocab_size+1}

    word2idx = tokenizer.word_index

    idx2word = {v:k for k,v in word2idx.items()}

    return word2idx, idx2word, tokenizer
# Declare Class Model

class POSTaggingModel(tf.keras.Model):

    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim,

                max_seqlen, rnn_output_dim, **kwargs):

        super(POSTaggingModel, self).__init__(**kwargs)

        self.embed = tf.keras.layers.Embedding(source_vocab_size, embedding_dim,

                                              input_length=max_seqlen)

        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)

        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_output_dim, 

                                                                    return_sequences=True))

        self.dense = tf.keras.layers.TimeDistributed(

        tf.keras.layers.Dense(target_vocab_size))

        self.activation = tf.keras.layers.Activation("softmax")

        

    def call(self, x):

        x = self.embed(x)

        x = self.dropout(x)

        x = self.rnn(x)

        x = self.dense(x)

        x = self.activation(x)

        return x

        

        
def masked_accuracy():

    def masked_accuracy_fn(ytrue, ypred):

        ytrue = tf.keras.backend.argmax(ytrue, axis=-1)

        ypred = tf.keras.backend.argmax(ypred, axis=-1)

        mask = tf.keras.backend.cast(

        tf.keras.backend.not_equal(ypred, 0), tf.int32)

        matches = tf.keras.backend.cast(

        tf.keras.backend.equal(ytrue, ypred), tf.int32) * mask

        numer = tf.keras.backend.sum(matches)

        denom = tf.keras.backend.maximum(tf.keras.backend.sum(mask), 1)

        accuracy = numer/denom

        return accuracy

    return masked_accuracy_fn
def clean_logs(data_dir):

    logs_dir = os.path.join(data_dir, "logs")

    shutil.rmtree(logs_dir, ignore_errors=True)

    return logs_dir
NUM_PAIRS = None

EMBEDDING_DIM = 128

RNN_OUTPUT_DIM = 256

BATCH_SIZE = 128

NUM_EPOCHS = 50
tf.random.set_seed(42)

data_dir="./data"

logs_dir = clean_logs(data_dir)
sents, poss = download_and_read("./datasets", num_pairs=NUM_PAIRS)

assert(len(sents) == len(poss))

print("# of records: {:d}".format(len(sents)))
word2idx_s, idx2word_s, tokenizer_s = preprocess_input(sents, vocab_size=9000)

word2idx_t, idx2word_t, tokenizer_t = preprocess_input(poss, vocab_size=38, lower=False)
source_vocab_size = len(word2idx_s)

target_vocab_size = len(word2idx_t)

print("vocab sizes (source): {:d}, (target) {:d}".format(source_vocab_size, target_vocab_size))
sequence_lengths = np.array([len(s.split()) for s in sents])

print([ (p, np.percentile(sequence_lengths,p)) for p in [75, 80, 85, 90, 95, 100] ])
max_seqlen = max(sequence_lengths)

max_seqlen
sentence_as_ints = tokenizer_s.texts_to_sequences(sents)

sentence_as_ints = tf.keras.preprocessing.sequence.pad_sequences(sentence_as_ints, 

                                                                maxlen = max_seqlen,

                                                                padding="post")

poss_as_ints = tokenizer_t.texts_to_sequences(poss)

poss_as_ints = tf.keras.preprocessing.sequence.pad_sequences(poss_as_ints, 

                                                                maxlen = max_seqlen,

                                                                padding="post")

dataset = tf.data.Dataset.from_tensor_slices((sentence_as_ints, poss_as_ints))

idx2word_s[0] , idx2word_t[0] ="PAD", "PAD"
poss_as_catints = []

for p in poss_as_ints:

    poss_as_catints.append(tf.keras.utils.to_categorical(p, num_classes=target_vocab_size,

                                                        dtype="int32"))

poss_as_catints = tf.keras.preprocessing.sequence.pad_sequences(poss_as_catints,

                                                               maxlen = max_seqlen)

dataset = tf.data.Dataset.from_tensor_slices((sentence_as_ints, poss_as_catints))
# split into train, test, and validation

batch_size = BATCH_SIZE

dataset = dataset.shuffle(10000)

test_size = len(sents)//3

val_size = (len(sents) - test_size)//10

test_dataset = dataset.take(test_size)

val_dataset = dataset.skip(test_size).take(val_size)

train_dataset = dataset.skip(test_size + val_size)



train_dataset = train_dataset.batch(batch_size)

val_dataset = val_dataset.batch(batch_size)

test_dataset = test_dataset.batch(batch_size)
embedding_dim = EMBEDDING_DIM

rnn_output_dim = RNN_OUTPUT_DIM
model = POSTaggingModel(source_vocab_size, target_vocab_size, 

                       embedding_dim, max_seqlen, rnn_output_dim)

model.build(input_shape=(batch_size, max_seqlen))

model.summary()

embedding_dim = EMBEDDING_DIM

rnn_output_dim = RNN_OUTPUT_DIM
model.compile(loss="categorical_crossentropy",

optimizer="adam",

metrics=["accuracy", masked_accuracy()])
tf.random.set_seed(42)

num_epochs = NUM_EPOCHS

best_model_file = os.path.join(data_dir, "best_model.h5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file, save_weights_only=True, save_best_only=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

history = model.fit(train_dataset,

                   epochs=num_epochs,

                   validation_data=val_dataset,

                   callbacks=[checkpoint, tensorboard])

# evaluate with test set

best_model = POSTaggingModel(source_vocab_size, target_vocab_size,

    embedding_dim, max_seqlen, rnn_output_dim)

best_model.build(input_shape=(batch_size, max_seqlen))

best_model.load_weights(best_model_file)

best_model.compile(

    loss="categorical_crossentropy",

    optimizer="adam", 

    metrics=["accuracy", masked_accuracy()])
test_loss, test_acc, test_masked_acc = best_model.evaluate(test_dataset)

print("test loss: {:.3f}, test accuracy: {:.3f}, masked test accuracy: {:.3f}".format(

    test_loss, test_acc, test_masked_acc))
labels, predictions = [], []

is_first_batch = True

accuracies = []
for test_batch in test_dataset:

    inputs_b, outputs_b = test_batch

    preds_b = best_model.predict(inputs_b)

    # convert from categorical to list of ints

    preds_b = np.argmax(preds_b, axis=-1)

    outputs_b = np.argmax(outputs_b.numpy(), axis=-1)

    for i, (pred_l, output_l) in enumerate(zip(preds_b, outputs_b)):

        assert(len(pred_l) == len(output_l))

        pad_len = np.nonzero(output_l)[0][0]

        acc = np.count_nonzero(

            np.equal(

                output_l[pad_len:], pred_l[pad_len:]

            )

        ) / len(output_l[pad_len:])

        accuracies.append(acc)

        if is_first_batch:

            words = [idx2word_s[x] for x in inputs_b.numpy()[i][pad_len:]]

            postags_l = [idx2word_t[x] for x in output_l[pad_len:] if x > 0]

            postags_p = [idx2word_t[x] for x in pred_l[pad_len:] if x > 0]

            print("labeled  : {:s}".format(" ".join(["{:s}/{:s}".format(w, p) 

                for (w, p) in zip(words, postags_l)])))

            print("predicted: {:s}".format(" ".join(["{:s}/{:s}".format(w, p) 

                for (w, p) in zip(words, postags_p)])))

            print(" ")

    is_first_batch = False
accuracy_score = np.mean(np.array(accuracies))

print("pos tagging accuracy: {:.3f}".format(accuracy_score))