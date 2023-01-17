import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

import time # kernels have a 2 hour limit

from collections import Counter
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
train.sample()
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt' 
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')  

# creates a mapping from the words to the embedding vectors=
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE)) 
def check_coverage(vocab,embeddings_index):
    a, oov, k, i = {}, {}, 0, 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print(f'Found embeddings for {(len(a) / len(vocab)):.2%} of vocab')
    print(f'Found embeddings for  {(k / (k + i)):.2%} of all text')
    sorted_x = sorted(oov.items(), key=(lambda x: x[1]), reverse=True)

    return sorted_x

def get_vocab(question_series):
    sentences = question_series.str.split().values #get a list of lists of words
    words = [item for sublist in sentences for item in sublist] # flatten list into just words
    return dict(Counter(words)) # count words
vocab = get_vocab(train["question_text"])
out_of_vocab = check_coverage(vocab, embeddings_index)
out_of_vocab[:10]
punct = set('?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~°√' + '“”’')
embed_punct = punct & set(embeddings_index.keys())

def clean_punctuation(txt):
    for p in "/-":
        txt = txt.replace(p, ' ')
    for p in "'`‘":
        txt = txt.replace(p, '')
    for p in punct:
        txt = txt.replace(p, f' {p} ' if p in embed_punct else ' _punct_ ') 
        #known punctuation gets space padded, otherwise we use a newn token
    return txt
train["question_text"] = train["question_text"].map(lambda x: clean_punctuation(x)).str.replace('\d+', ' # ')
test["question_text"] = test["question_text"].map(lambda x: clean_punctuation(x)).str.replace('\d+', ' # ')
vocab = get_vocab(train["question_text"])
out_of_vocab = check_coverage(vocab, embeddings_index)
out_of_vocab[:10]
x = train["question_text"].str.split().map(lambda x: len(x))
x.describe()
train, validation = train_test_split(train, test_size=0.08, random_state=20181224)

embed_size = 300 #word vector sizes
vocab_size = 95000 # words in vocabulary
maxlen = 100 # max words to use per question

# fill up the missing values
train_X = train["question_text"].fillna("_##_").values
val_X = validation["question_text"].fillna("_##_").values
test_X = test["question_text"].fillna("_##_").values

# Use Keras to tokenize and pad sequences
tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=False)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

# Get the response
train_y = train['target'].values
val_y = validation['target'].values
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # only want at most vocab_size words in our vocabulary 
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #first, make our embedding matric random7
num_missed = 0
for word, i in word_index.items(): # insert embeddings we that exist into our matrix
    if i >= vocab_size: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    else: num_missed += 1
print(num_missed)
BATCH_SIZE = 64

# mmakes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# should be batchsize x length of each question (vectors of numbers representing indices into the embedding matrix)
X = tf.placeholder(tf.int32, [None, maxlen], name='X')

# 1d vector with size = None because we want to predict one val for each q, but want variable batch sizes
Y = tf.placeholder(tf.float32, [None], name='Y')
batch_size = tf.placeholder(tf.int64, name='batch_size')
#training = tf.placeholder(tf.bool, name='training')
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=1000).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size) #this one does not shuffle

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes) 

# To choose which dataset we use, we simply initialize the appropriate one using the init_ops below
train_init_op = iterator.make_initializer(dataset)
test_init_op = iterator.make_initializer(test_dataset)

questions, labels = iterator.get_next()
embeddings = tf.get_variable(name="embeddings", shape=embedding_matrix.shape,
                             initializer=tf.constant_initializer(np.array(embedding_matrix)), 
                             trainable=False)
embed = tf.nn.embedding_lookup(embeddings, questions)
l2_amt = 0.0
regularizer = tf.contrib.layers.l2_regularizer(l2_amt)
Q = tf.nn.elu(tf.layers.conv1d(embed, embed_size, 3, padding='SAME', kernel_regularizer=regularizer))
K = tf.nn.elu(tf.layers.conv1d(embed, embed_size, 3, padding='SAME', kernel_regularizer=regularizer))
V = tf.nn.elu(tf.layers.conv1d(embed, embed_size, 3, padding='SAME', kernel_regularizer=regularizer))
QKT = tf.matmul(Q, K, transpose_b=True)/np.sqrt(embed_size)

#softmax manual computation
exp_QKT = tf.exp(QKT)
qkt_sum = tf.reduce_sum(exp_QKT, axis=[1,2])
softmax = exp_QKT/tf.expand_dims(tf.expand_dims(qkt_sum, 1), 1)

attention = tf.matmul(softmax, V)
kernel_lens = [3, 4, 5, 6, 7] # (3, 4, 5) and (7, 7, 7) worked well in the paper
num_filters = len(kernel_lens)

Z = [tf.layers.conv1d(attention, 100, ker_len, padding='SAME', kernel_regularizer=regularizer) 
     for ker_len in kernel_lens]
#Z = [tf.nn.conv1d(embed, W[i], stride = 1, padding = 'SAME') for i in range (num_filters) ]
A = [tf.nn.relu(Z[i]) for i in range (num_filters)]
P = [tf.reduce_max(A[i], axis=2) for i in range (num_filters)]

FLAT = tf.contrib.layers.flatten(tf.concat(P, axis=1))

last_layer = tf.layers.dense(FLAT, 1, kernel_regularizer=regularizer) #fully connected layer
prediction = tf.nn.sigmoid(last_layer) #activation function
prediction = tf.squeeze(prediction, [1]) # layers.dense returns a tensor, but we want to remove the extra dimension
learning_rate=0.01

l2_loss = tf.losses.get_regularization_loss()

# define cross entropy loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(last_layer), labels=labels)
loss = tf.reduce_mean(loss) + l2_loss

# define our optimizer to minimize the loss
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss) #MomentumOptimizer
with tf.name_scope('metrics'):
    F1, f1_update = tf.contrib.metrics.f1_score(labels=labels, predictions=prediction, name='my_metric')
    
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
reset_op = tf.variables_initializer(var_list=running_vars)
num_epochs = 10
seed = 3 # we use a seed to have deterministic results

sess = tf.Session()

# Run the initialization
sess.run(tf.global_variables_initializer()) # initializes all of our variables
sess.run(tf.local_variables_initializer()) # need this for the f1 metric to work

costs, f1, val_f1 = [], [], []
def check_validation(sz = 90, plot=False):
    sess.run(test_init_op, feed_dict={X: val_X, Y: val_y, batch_size: sz})
    val_pred = np.concatenate([sess.run(prediction) for _ in range(int(val_X.shape[0]/sz))])
    thresholds = [i/200 for i in range(10, 120, 1)] 
    scores = [metrics.f1_score(val_y,np.int16(val_pred > t)) for t in thresholds]
    
    if plot:
        plt.plot(thresholds, scores)
        plt.ylabel("F1 Score")
        plt.xlabel('Threshold')
        plt.title("F1 Score by thresholds for Validation Set")
        plt.show()
        
    return max(scores), thresholds[np.argmax(scores)]
## Want to train for 6600 seconds to stay under the time limit of 7200 seconds (2 hours)
start = time.time()
end = 0
max_time = 6700

# initialize iterator with train data
sess.run(train_init_op, feed_dict={X:train_X, Y:train_y, batch_size:BATCH_SIZE})
num_iter = 2000 # print after each num_iter
num_batches = int(train_X.shape[0] / BATCH_SIZE) # number of batches/minibatches

# Training Loop
for epoch in range(1, num_epochs+1):
    seed += seed # want a different random shuffle every time, but still have deterministic results
    tf.set_random_seed(seed)
    iter_cost = 0.
    
    # the last batch is smaller than the rest, so we will use 
    # this to keep track of the number of iterations to get the right average cost
    prev_iter = 0. 
    
    for i in range(num_batches):
        _ , batch_loss, _ = sess.run([optimizer, loss, f1_update]) 
        iter_cost += batch_loss
        
        # End training after 
        end = time.time()
        if (end-start > max_time): 
            break
        
        if (i % num_iter == 0 and i > 0): 
            iter_cost /= (i-prev_iter) # get average batch cost
            prev_iter = i #update prev_iter for next iteration
            cur_f1 = sess.run(F1)
            sess.run(reset_op) # reset counters for F1
            
            f1.append(cur_f1)
            costs.append(iter_cost)
            print (f"Epoch {epoch} Iteration {i:5}  cost: {iter_cost:.6f}  f1: {cur_f1:.5f}  time: {end-start:4.4f}")
            batch_cost = 0. #reset batch_cost
            
    val_f1.append(check_validation()[0])
    print(f'val f1 {val_f1[-1]}')
    sess.run(train_init_op, feed_dict={X:train_X, Y:train_y, batch_size:BATCH_SIZE})
    if (end-start > max_time): 
        break
def easy_plot(yvals, ylabel='', num_iter=num_iter):
    plt.plot(yvals)
    plt.ylabel(ylabel)
    plt.xlabel(f'Iterations (per {num_iter})')
    plt.title(f"{ylabel} by Iterations for Learning Rate = {learning_rate}")
    plt.show()
    
easy_plot(np.squeeze(costs), 'Cost')
easy_plot(np.squeeze(f1), 'Train F1 Score')
easy_plot(np.squeeze(val_f1), 'Validation F1 Score', num_iter=20000)
tf.set_random_seed(2018)
score, thresh = check_validation(plot=True)
print(f"Best Validation F1 Score is {score:.4f} at threshold {thresh}")
sz=30
temp_y = val_y[:test_X.shape[0]]
sub = test[['qid']]
sess.run(test_init_op, feed_dict={X: test_X, Y: temp_y, batch_size:sz})
sub['prediction'] = np.concatenate([sess.run(prediction) for _ in range(int(test_X.shape[0]/sz))])
sub['prediction'] = (sub['prediction'] > thresh).astype(np.int16)
sub.to_csv("submission.csv", index=False)
sub.sample()