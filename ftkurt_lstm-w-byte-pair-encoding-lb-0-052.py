import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from keras.models import Model

from keras.layers import Dense, Embedding, Input

from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D

from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding

from keras.layers.merge import Concatenate

from keras.preprocessing import text, sequence

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# Source: https://raw.githubusercontent.com/soaxelbrooke/python-bpe/master/bpe/encoder.py

# coding=utf-8

""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """

from collections import Counter



try:

    from typing import Dict, Iterable, Callable, List, Any, Iterator

except ImportError:

    pass



from nltk.tokenize import wordpunct_tokenize

from tqdm import tqdm

import toolz

import json



DEFAULT_EOW = '__eow'

DEFAULT_SOW = '__sow'

DEFAULT_UNK = '__unk'

DEFAULT_PAD = '__pad'





class Encoder:

    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.

    """



    def __init__(self, vocab_size=8192, pct_bpe=0.2, word_tokenizer=None,

                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None, strict=False, 

                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):

        if vocab_size < 1:

            raise ValueError('vocab size must be greater than 0.')



        self.EOW = EOW

        self.SOW = SOW

        self.eow_len = len(EOW)

        self.sow_len = len(SOW)

        self.UNK = UNK

        self.PAD = PAD

        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))

        self.vocab_size = vocab_size

        self.pct_bpe = pct_bpe

        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])

        self.bpe_vocab_size = vocab_size - self.word_vocab_size

        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize

        self.custom_tokenizer = word_tokenizer is not None

        self.word_vocab = {}  # type: Dict[str, int]

        self.bpe_vocab = {}  # type: Dict[str, int]

        self.inverse_word_vocab = {}  # type: Dict[int, str]

        self.inverse_bpe_vocab = {}  # type: Dict[int, str]

        self._progress_bar = iter if silent else tqdm

        self.ngram_min = ngram_min

        self.ngram_max = ngram_max

        self.strict = strict



    def mute(self):

        """ Turn on silent mode """

        self._progress_bar = iter



    def unmute(self):

        """ Turn off silent mode """

        self._progress_bar = tqdm



    def byte_pair_counts(self, words):

        # type: (Encoder, Iterable[str]) -> Iterable[Counter]

        """ Counts space separated token character pairs:

            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}

        """

        for token, count in self._progress_bar(self.count_tokens(words).items()):

            bp_counts = Counter()  # type: Counter

            for ngram in token.split(' '):

                bp_counts[ngram] += count

            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):

                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]



                for ngram in ngrams:

                    bp_counts[''.join(ngram)] += count



            yield bp_counts



    def count_tokens(self, words):

        # type: (Encoder, Iterable[str]) -> Dict[str, int]

        """ Count tokens into a BPE vocab """

        token_counts = Counter(self._progress_bar(words))

        return {' '.join(token): count for token, count in token_counts.items()}



    def learn_word_vocab(self, sentences):

        # type: (Encoder, Iterable[str]) -> Dict[str, int]

        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """

        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))

        for token in set(self.required_tokens or []):

            word_counts[token] = int(2**63)

        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])

        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}



    def learn_bpe_vocab(self, words):

        # type: (Encoder, Iterable[str]) -> Dict[str, int]

        """ Learns a vocab of byte pair encodings """

        vocab = Counter()  # type: Counter

        for token in {self.SOW, self.EOW}:

            vocab[token] = int(2**63)

        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):

            for byte_pair, count in byte_pair_count.items():

                vocab[byte_pair] += count



            if (idx + 1) % 10000 == 0:

                self.trim_vocab(10 * self.bpe_vocab_size, vocab)



        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]

        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}



    def fit(self, text):

        # type: (Encoder, Iterable[str]) -> None

        """ Learn vocab from text. """

        _text = [l.lower().strip() for l in text]



        # First, learn word vocab

        self.word_vocab = self.learn_word_vocab(_text)



        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))

                           if word not in self.word_vocab]

        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)



        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}

        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}



    @staticmethod

    def trim_vocab(n, vocab):

        # type: (int, Dict[str, int]) -> None

        """  Deletes all pairs below 10 * vocab size to prevent memory problems """

        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])

        pairs_to_trim = [pair for pair, count in pair_counts[n:]]

        for pair in pairs_to_trim:

            del vocab[pair]



    def subword_tokenize(self, word):

        # type: (Encoder, str) -> List[str]

        """ Tokenizes inside an unknown token using BPE """

        end_idx = min([len(word), self.ngram_max])

        sw_tokens = [self.SOW]

        start_idx = 0



        while start_idx < len(word):

            subword = word[start_idx:end_idx]

            if subword in self.bpe_vocab:

                sw_tokens.append(subword)

                start_idx = end_idx

                end_idx = min([len(word), start_idx + self.ngram_max])

            elif len(subword) == 1:

                sw_tokens.append(self.UNK)

                start_idx = end_idx

                end_idx = min([len(word), start_idx + self.ngram_max])

            else:

                end_idx -= 1



        sw_tokens.append(self.EOW)

        return sw_tokens



    def tokenize(self, sentence):

        # type: (Encoder, str) -> List[str]

        """ Split a sentence into word and subword tokens """

        word_tokens = self.word_tokenizer(sentence.lower().strip())



        tokens = []

        for word_token in word_tokens:

            if word_token in self.word_vocab:

                tokens.append(word_token)

            else:

                tokens.extend(self.subword_tokenize(word_token))



        return tokens

    def transform(self, sentences, reverse=False, fixed_length=None):

        # type: (Encoder, Iterable[str], bool, int) -> Iterable[List[int]]

        """ Turns space separated tokens into vocab idxs """

        direction = -1 if reverse else 1

        for sentence in self._progress_bar(sentences):

            encoded = []

            tokens = list(self.tokenize(sentence.lower().strip()))

            for token in tokens:

                if token in self.word_vocab:

                    encoded.append(self.word_vocab[token])

                elif token in self.bpe_vocab:

                    encoded.append(self.bpe_vocab[token])

                else:

                    encoded.append(self.word_vocab[self.UNK])



            if fixed_length is not None:

                encoded = encoded[:fixed_length]

                while len(encoded) < fixed_length:

                    encoded.append(self.word_vocab[self.PAD])



            yield encoded[::direction]
import matplotlib.pyplot as plt

from IPython.display import clear_output
%matplotlib inline

pd.set_option('display.max_colwidth', -1)

plt.rcParams['figure.figsize'] = [15,9]
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Lengths Train/Test: {} / {}".format(len(train),len(test)))
train = train.sample(frac=1)
train.head(2)
def fit_encoder(encoder,corpus):

    start = time.time()

    encoder.fit(corpus)

    print("Encoder trained: "+str(int(time.time() - start))+"s")
list_sentences_train = train["comment_text"].fillna("__empty__").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_test = test["comment_text"].fillna("__empty__").values
vocabulary = 50000

encoder = Encoder(vocabulary, ngram_max=10)

corpus = list_sentences_train

fit_encoder(encoder,corpus)
list_tokenized_train = list(encoder.transform(list_sentences_train))

list_tokenized_test = list(encoder.transform(list_sentences_test))
cutout_percentile = 99.0
sizes = [len(x) for x in list_tokenized_train+list_tokenized_test]

max_size = int(np.percentile(sizes,cutout_percentile))

max_features = max([max(x) for x in list_tokenized_train+list_tokenized_test])+1
print("Vocabulary: {}, Max Entry Size: {}".format(max_features,max_size))
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=max_size)

X_te = sequence.pad_sequences(list_tokenized_test, maxlen=max_size)
class PlotModel(Callback):

    def set_params_(self, model_type, checkpoint_path, dataset, 

                   batch_size, plot_per_batch,

                   dictionary_size, max_review_length, 

                   best_model_monitor, early_stop_monitor):

        self.path = checkpoint_path

        self.model_type = model_type

        self.dataset = dataset

        self.batch_size = batch_size

        self.dictionary_size = dictionary_size

        self.max_review_length = max_review_length

        self.best_model_monitor = best_model_monitor

        self.early_stop_monitor = early_stop_monitor

        self.plot_per_batch = plot_per_batch

    

    def get_monitor_ticks(self):

        if self.best_model_monitor == "val_acc":

            bm_ind = np.argmax(self.val_acc)

        else:

            bm_ind = np.argmin(self.val_losses)

        

        if self.early_stop_monitor == "val_acc":

            es_ind = np.argmax(self.val_acc)

        else:

            es_ind = np.argmin(self.val_losses)

        return [bm_ind, es_ind]

    

    def plot(self, save):

        clear_output(wait=True)

        fig = plt.figure()

        plt.grid(True)

        plt.plot(self.x, self.acc, '--', label="acc")

        plt.plot(self.x, self.val_acc, label="val_acc")

        plt.plot(self.x, self.losses, '--', label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        bm_ind, es_ind = self.get_monitor_ticks()

        plt.axvline(x=self.x[bm_ind],alpha=0.6,color="green", linestyle='--')

        plt.axvline(x=self.x[es_ind],alpha=0.6,color="red", linestyle='--')

        title = "{}: accuracy & loss for {} (length:{}, val acc/loss: {:.4f}/{:.4f})"

        title = title.format(self.model_type,self.dataset,self.max_review_length,max(self.val_acc),min(self.val_losses))

        plt.title(title)

        plt.legend(['acc: train', 'acc: validation','loss: train', 'loss: validation'], loc='upper left')

        plt.show()

        if save:

            fig.savefig(self.path+"-figure.png")

            

    def on_train_begin(self, logs={}):

        self.i = 1

        self.x = [0]

        self.acc = [0]

        self.val_acc = [0]

        self.losses = [1]

        self.val_losses = [1]

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        if 'acc' in logs  and 'loss' in logs:

            self.logs.append(logs)

            self.x.append(self.i)

            self.acc.append(logs.get('acc'))

            self.losses.append(logs.get('loss'))

            if "val_acc" in logs and "val_loss" in logs:

                self.val_acc.append(logs.get('val_acc'))

                self.val_losses.append(logs.get('val_loss'))

            else:

                self.val_acc.append(self.val_acc[len(self.val_acc)-1])

                self.val_losses.append(self.val_losses[len(self.val_losses)-1])

            self.i += 1

            self.plot(False)

        else:

            keys = []

            for k in ['acc', 'val_acc', 'loss', 'val_loss']:

                if k not in logs:

                    keys.append(k)

            print(("Missing parameters",keys,logs))

    

    def on_batch_end(self, batch, logs={}):

        if batch%self.plot_per_batch != 0:

            return

        else:

            self.on_epoch_end(batch, logs)

    

    def on_train_end(self, epoch, logs={}):

        self.plot(True)
def get_lstm():

    embed_size = 128

    inp = Input(shape=(max_size, ))

    x = Embedding(max_features, embed_size)(inp)

    x = Bidirectional(LSTM(50, return_sequences=True))(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model
model_type = "lstm"
model = get_lstm()

batch_size = 256

epochs = 1

es_patience = 1

file_path=model_type+".weights_base.best.hdf5"
model.summary()
number_of_plots = 200

number_of_batches_per_epoch = len(X_t)/batch_size

plot_per_batch = int(number_of_batches_per_epoch/(number_of_plots/epochs))

if plot_per_batch == 0:

    plot_per_batch = 1

plot_per_batch
plotter = PlotModel()

plotter.set_params_(model_type, model_type+"-checkpoint-path", "bpe-{}k".format(int(vocabulary/1000)), 

                   batch_size, plot_per_batch,

                   vocabulary, max_size, 

                   'val_loss', 'val_loss')



checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

earlyStopping = EarlyStopping(monitor="val_loss", mode="max", patience=es_patience)

callbacks_list = [plotter, checkpoint, earlyStopping]
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
model = get_lstm()

model.load_weights(file_path)
sample_per_prediction = 1000

def get_predictions(model,data):

    preds = []

    cnt = 0

    total = len(data)

    while cnt*sample_per_prediction<len(data):

        start = time.time()

        sample = data[cnt*sample_per_prediction:(cnt+1)*sample_per_prediction]

        preds += model.predict(sample).tolist()

        clear_output(wait=True)

        cnt += 1

        percent = cnt*sample_per_prediction/total

        print("Progress: {:.2f}%, (Remaining: {}s)".format(100.0*percent,int(total*(1-percent)*int(time.time() - start)/sample_per_prediction)))

    return np.array(preds)
y_test = get_predictions(model,X_te)
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_classes] = y_test
def plot_predictions_grid(df):

    fig = plt.figure()

    for i,c in enumerate(list_classes):

        sub1 = fig.add_subplot(321+i)

        sub1.hist(df[c].values, label=c, bins=100, color="red")

        sub1.set_title(c+" Prediction Distribution")

        sub1.set_yscale('log', nonposy='clip')

        sub1.legend(loc='upper right')

    plt.show()
plot_predictions_grid(sample_submission)
sample_submission.to_csv("baseline.csv", index=False)