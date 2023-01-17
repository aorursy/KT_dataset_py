SEED = 1337

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] =  "0"

import time
import numpy as np
import tensorflow as tf
import pandas as pd
import itertools as it
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

np.random.seed(SEED)
tf.set_random_seed(SEED)
## Text Cleaning
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_text(x):
    for dic in [contraction_mapping, mispell_dict, punct_mapping]:
        for word in dic.keys():
            x = x.replace(word, dic[word])
    return x

## Loading and preprocessing text
def load_and_preprocess_data(max_features=50000, maxlen=70):
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    #print("Train shape : ",train_df.shape)
    #print("Test shape : ",test_df.shape)

    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

    train_df['question_text'] = train_df['question_text'].fillna("").apply(lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].fillna("").apply(lambda x: clean_text(x))

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X) + list(test_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values

    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))

    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]

    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
# Generator for dataset
def data_gen(x, y, batch_size=32, shuffle=True):
        x = np.array(x)
        if len(x) == len(y) : y = np.array(y)

        data_size = len(x)
        num_batches_per_epoch = int((len(x)-1)/batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x = x[shuffle_indices]
            if len(x) == len(y): y = y[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            bat_x = x[start_index:end_index]
            bat_y = y[start_index:end_index]
            bat_y = np.reshape(bat_y, (-1, 1))
            yield np.array(bat_x), np.array(bat_y)
## TensorFlow BiRNN Model
class RNN:
    def __init__(self,
                num_classes=1,
                learning_rate=0.001,
                batch_size=None,
                seq_length=70,
                vocab_size=10000,
                embed_size=300,
                hidden_size=64,
                training=True):

        ## Hyperparamters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.training = True
        self.decay_steps = 1000
        self.decay_rate = 0.95

        ## Placeholders
        self.input_x = tf.placeholder(tf.int64, shape=[self.batch_size, self.seq_length], name="question")
        self.input_y = tf.placeholder(tf.int64, shape=[self.batch_size, self.num_classes], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ## Model Checkpoint
        self.model_dir = "weights"
        self.model_name = "rnn.ckpt"

        ## Embedding Weight
        with tf.name_scope("embedding"):
            self.embedding_matrix = tf.get_variable("embedding_matrix", shape=[self.vocab_size, self.embed_size], initializer=tf.random_normal_initializer(stddev=0.1))

        ##
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.logits = self.inference()
        self.predictions = tf.nn.sigmoid(self.logits)
        self.loss_val = self.loss()
        self.optimizer_op = self.optimizer().minimize(self.loss_val, global_step=self.global_step)
        correct_predictions = tf.equal(tf.cast(tf.round(self.predictions), tf.int64), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        _, self.f1 = tf.contrib.metrics.f1_score(self.input_y, self.predictions)

    def inference(self):
        """ Embedding """
        self.embedding_words = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

        """ BiGRU """
        fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        
        #fw_cell = tf.contrib.cudnn_rnn.CudnnGRU(1, self.hidden_size, self.hidden_size)
        #bw_cell = tf.contrib.cudnn_rnn.CudnnGRU(1, self.hidden_size, self.hidden_size)

        if self.dropout_keep_prob is not None:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedding_words, dtype=tf.float32)
        output_rnn = tf.concat(outputs, axis=2)
        final_output = output_rnn[:, -1, :]
        
        x = tf.layers.dense(final_output, self.hidden_size)
        logits = tf.layers.dense(x, self.num_classes)
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32), logits=self.logits))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_loss
        return loss

    def optimizer(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate)
        return train_op

    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = os.path.join(self.model_dir, self.model_name)
        saver.save(sess, save_path, global_step=self.global_step)
        #print("Session saved.")

    def restore_model(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(self.model_dir, ckpt_name))
            print("Session restored.")

    def train_once(self, sess, train_data):
        loss = 0
        accuracy = 0
        f1 = 0
        n_batch = 0

        while True:
            try:
                x, y = next(train_data)
                feed_data = {self.input_x: x, self.input_y: y, self.dropout_keep_prob: 1.0}
                _, l, a, f = sess.run([self.optimizer_op, self.loss_val, self.accuracy, self.f1], feed_dict=feed_data)
                loss += l
                accuracy += a
                f1 += f
                n_batch += 1
            except StopIteration as e:
                break

        if n_batch != 0:
            loss = loss/n_batch
            accuracy = accuracy/n_batch
            f1 = f1/n_batch

        return [loss, accuracy, f1]

    def eval_once(self, sess, eval_data):
        loss = 0
        accuracy = 0
        f1 = 0
        n_batch = 0

        while True:
            try:
                x, y = next(eval_data)
                feed_data = {self.input_x: x, self.input_y: y, self.dropout_keep_prob: 1.0}
                l, a, f = sess.run([self.loss_val, self.accuracy, self.f1], feed_dict=feed_data)
                loss += l
                accuracy += a
                f1 += f
                n_batch += 1
            except StopIteration as e:
                break

        if n_batch != 0:
            loss = loss/n_batch
            accuracy = accuracy/n_batch
            f1 = f1/n_batch

        return [loss, accuracy, f1]

    def train(self, train_data, valid_data, epochs=5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ## Restore Session
            self.restore_model(sess)

            train_data_copy = list(it.tee(train_data, epochs))
            valid_data_copy = list(it.tee(valid_data, epochs))

            print("Epoch\tTime\tTrain Loss\tTrain Acc\tTrain F1\tVal Loss\tVal Acc\t\tVal F1")
            print("="*110)

            for epoch in range(epochs):
                start_time = time.time()
                train_loss, train_acc, train_f1 = self.train_once(sess, train_data_copy[epoch])
                valid_loss, valid_acc, valid_f1 = self.eval_once(sess, valid_data_copy[epoch])
                time_taken = time.time() - start_time

                # print("Epoch: {:2d} - {:3.4f} - Loss: {:1.5f} - Acc: {:0.5f} - Val loss: {:1.5f} - Val acc: {:0.5f}".
                #     format(epoch, time_taken, train_loss, train_acc, valid_loss, valid_acc))
                print("{:2d}\t{:3.2f}\t{:1.8f}\t{:0.8f}\t{:0.8f}\t{:1.8f}\t{:0.8f}\t{:0.8f}".
                    format(epoch+1, time_taken, train_loss, train_acc, train_f1, valid_loss, valid_acc, valid_f1))

                ## Save Session
                self.save_model(sess)
                
    def predict(self, test_data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_time = time.time()
            test_predict = np.array([])

            while True:
                try:
                    x, y = next(test_data)
                    feed_data = {self.input_x: x, self.dropout_keep_prob: 1.0}
                    r = sess.run([self.predictions], feed_dict=feed_data)
                    test_predict = np.append(test_predict, r)
                except StopIteration as e:
                    break

            time_taken = time.time() - start_time
            print("Time Taken: {:2.5f}".format(time_taken))

        return test_predict
start_time = time.time()
# Hyperparamters
num_classes=1
learning_rate=0.05
batch_size=512
seq_length=40
vocab_size=95000
embed_size=300
hidden_size=64
training=True
epochs=10
print("[+]Preprocessing text...")
train_X, valid_X, test_X, train_y, valid_y, word_index = load_and_preprocess_data(max_features=vocab_size, maxlen=seq_length)

print("[+]Preparing generators...")
train_gen = data_gen(train_X, train_y, batch_size=batch_size)
valid_gen = data_gen(valid_X, valid_y, batch_size=batch_size)
print("[+]TensorFlow model...")
rnn = RNN(
        num_classes=num_classes,
        learning_rate=learning_rate,
        batch_size=None,
        seq_length=seq_length,
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        training=training)

print("[+]Training...")
time_taken = time.time() - start_time
print("Time Taken: {:2.5f}".format(time_taken))
rnn.train(train_gen, valid_gen, epochs=epochs)
test_gen = data_gen(test_X, test_X, batch_size=1024, shuffle=False)
test_predict = rnn.predict(test_gen)
pred_test_y = (test_predict > 0.5).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(out_df)
