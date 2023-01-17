# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#coding:utf-8
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
import time
from sklearn.metrics import f1_score
import sys
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
class BaseModel(object):
	def __init__(self):
		self.output_feeds = []
		print('init Base model')
		# pass
	def add_train_op(self, loss,lr_method='adam', lr=0.001,clip=-1):
		"""Defines self.train_op that performs an update on a batch
		Args:
			lr_method: (string) sgd method, for example "adam"
			lr: (tf.placeholder) tf.float32, learning rate
			loss: (tensor) tf.float32 loss to minimize
			clip: (python float) clipping of gradient. If < 0, no clipping
		"""
		_lr_m = lr_method.lower()  # lower to make sure

		with tf.variable_scope("train_step"):
			if _lr_m == 'adam':  # sgd method
				print('adam')
				optimizer = tf.train.AdamOptimizer(lr)
			elif _lr_m == 'adagrad':
				optimizer = tf.train.AdagradOptimizer(lr)
			elif _lr_m == 'sgd':
				optimizer = tf.train.GradientDescentOptimizer(lr)
			elif _lr_m == 'rmsprop':
				optimizer = tf.train.RMSPropOptimizer(lr)
			else:
				raise NotImplementedError("Unknown method {}".format(_lr_m))

			if clip > 0:  # gradient clipping if clip is positive
				grads, vs = zip(*optimizer.compute_gradients(loss))
				grads, gnorm = tf.clip_by_global_norm(grads, clip)
				self.train_op = optimizer.apply_gradients(zip(grads, vs))
			else:
				self.train_op = optimizer.minimize(loss)
		self.saver = tf.train.Saver(max_to_keep=3, pad_step_number=True)
		self.params = tf.trainable_variables()
	#check params
class TextCNN(BaseModel):
	def __init__(self,config):
		self.max_len = config["max_len"]
		self.vocab_size = config["vocab_size"]
		self.embedding_size = config["embedding_size"]
		self.n_class = config["n_class"]
		self.learning_rate = config["learning_rate"]

		# placeholder
		self.x = tf.placeholder(tf.int32, [None, self.max_len])
		self.label = tf.placeholder(tf.int32, [None])
		self.use_pretrain_embedding = False
		self.filter_sizes = config['filter_sizes']
		self.num_filters = config['num_filters']
		self.keep_prob = tf.placeholder(tf.float32)

	def build_graph(self):
		if not self.use_pretrain_embedding:
			embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
										 trainable=True)
		else:
			pass
		batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
		embeded_char_expanded = tf.expand_dims(batch_embedded,axis=-1)

		pool_list = list()
		for filter_size,filter_num in zip(self.filter_sizes,self.num_filters):
			with tf.variable_scope('cov2d-maxpool%s'% filter_size):
				filter_shape = [filter_size,300,1,filter_num]
				W = tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=0.1),name='W')
				b = tf.Variable(tf.constant(0.1,shape=[filter_num]),name='b')

				conv = tf.nn.conv2d(embeded_char_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')

				h = tf.nn.tanh(tf.nn.bias_add(conv,b),name='tanh')
				pooled = tf.nn.max_pool(
					h,
					ksize=[1,self.max_len-filter_size+1,1,1],
					strides=[1,1,1,1],
					padding='VALID',
					name='pool'
				)
				pool_list.append(pooled)
		total_filter_num = sum(self.num_filters)
		self.h_pool = tf.concat(pool_list,3)
		self.h_pool_flat = tf.reshape(self.h_pool,[-1,total_filter_num])
		self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

		logits = tf.layers.dense(self.h_pool_flat,self.n_class,activation=tf.nn.sigmoid)

		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))

		self.add_train_op(self.loss)

		self.prediction = tf.argmax(tf.nn.softmax(logits),axis=1)
		self.scores = tf.nn.softmax(logits)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
names = ["qid", "question_text", "target"]
import pandas as pd


def load_data(file_name, sample_ratio=1, names=names):
	'''load data from .csv file'''
	csv_file = pd.read_csv(file_name, names=names)
	shuffle_csv = csv_file.sample(frac=sample_ratio)
	return shuffle_csv["question_text"], shuffle_csv["target"]


def data_preprocessing_v2(all_text,train, max_len, max_words=40000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(all_text)
    train_idx = tokenizer.texts_to_sequences(train)
    train_padded = pad_sequences(train_idx, maxlen=max_len )
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, max_words + 2
def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_train, x_dev,y_train, y_dev = train_test_split(x_test, y_test, train_size=0.95,
	                                                  random_state=233)
    return x_train, x_dev, y_train, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch

def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 0.5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    scores,prediction = sess.run([model.scores,model.prediction], feed_dict)
    return scores,prediction


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
import os 
dataset_dir = '../input/'
print(os.listdir(dataset_dir))
EMBEDDING_FILE = dataset_dir+'glove.840B.300d.txt'
names = ["qid", "question_text", "target"]
data_dir =''
train=pd.read_csv(dataset_dir+'train.csv')
test=pd.read_csv(dataset_dir+'test.csv')
train_sam=train.sample(frac=1.0)
x_test=test['question_text']
x_train, y_train = train_sam['question_text'],train_sam['target']
max_features = 40000
max_len = 50
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_test.values)+list(x_train.values))
train_idx = tokenizer.texts_to_sequences(x_train)

X_train = pad_sequences(train_idx, maxlen=max_len)

test_idx = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(test_idx,maxlen=max_len)

print(X_train.shape)
print(X_test.shape)

	#x_train, vocab_size = data_preprocessing_v2(list(x_test.values)+list(x_train.values),x_train, max_len=50)
x_train, x_dev, y_train, y_dev, dev_size, train_size = split_dataset(X_train, y_train, 0.05)
	#x_train, x_dev,y_train, y_dev = train_test_split(X_train, y_train, train_size=0.95,
	#	                                                          random_state=233)


# 	def get_coefs(word, *arr):
# 		return word, np.asarray(arr, dtype='float32')


# 	embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
# 	embed_size = 300
# 	word_index = tokenizer.word_index
# 	nb_words = min(max_features, len(word_index))
# 	embedding_matrix = np.zeros((nb_words, embed_size))
# 	for word, i in word_index.items():
# 		if i >= max_features: continue
# 		embedding_vector = embeddings_index.get(word)
# 		if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print("loading embedding done !")
vocab_size = max_features

config = {
    "max_len": 50,
    "hidden_size": 64,
    "vocab_size": vocab_size,
    "embedding_size": 128,
    "n_class": 2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "train_epoch": 3
}
tf.reset_default_graph()
#classifier = ABLSTM(config)
config = {
    "max_len": 50,
    "vocab_size": vocab_size,
    "embedding_size": 300,
    "learning_rate": 1e-3,
    "l2_reg_lambda": 1e-3,
    "batch_size": 256,
    "n_class": 2,

    # random setting, may need fine-tune
    "filter_sizes": [1, 2, 3,5],
    "num_filters": [36, 36,36,36],
    "train_epoch": 6,
    "hidden_size": 256
}
# from text_cnn import TextCNN
# from bilstm import BLSTM
#classifier = BLSTM(config)
classifier = TextCNN(config)
classifier.build_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dev_batch = (x_dev, y_dev)
start = time.time()
predictions = []
pred_thresh =0.0 
for e in range(config["train_epoch"]):
    t0 = time.time()
    print("Epoch %d start !" % (e + 1))
    for x_batch, y_batch in tqdm(fill_feed_dict(x_train, y_train, config["batch_size"])):
        return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
        # attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
    t1 = time.time()
    print("Train Epoch time:  %.3f s" % (t1 - t0))
    scores,dev_preds = run_eval_step(classifier, sess, dev_batch)
    best_score = 0.0
    best_thresh =0.0
    for thresh in np.arange(0.1,0.501,0.01):
        score = f1_score(y_dev.values,(scores[:,1]>thresh))
        if score>best_score: 
            best_score = score
            best_thresh = thresh
            pred_thresh = best_thresh
    f1_scores = f1_score(y_dev.values,(scores[:,1]>best_thresh).astype(np.int))#,dev_preds = run_eval_step(classifier, sess, dev_batch)
    print("validation F1: %.3f " % f1_scores)
print("prediting on test_data")
print(pred_thresh)
target = [0 for i in range(test.shape[0])]
test['target'] = target
test_batch=(X_test,test['target'].values)
scores,dev_preds = run_eval_step(classifier,sess,test_batch)
preditions = (scores[:,1]>pred_thresh).astype(np.int)
submission_final = pd.DataFrame.from_dict({'qid':test['qid']})
submission_final['prediction'] =preditions
submission_final.to_csv('submission.csv',index = False)
submission_final.head(20)
