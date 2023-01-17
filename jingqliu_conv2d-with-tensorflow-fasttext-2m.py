import pandas as pd
import numpy as np
import re
import csv
import tensorflow as tf
import nltk
import gc
from gensim.models import Word2Vec
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from collections import Counter
#This dataset is from Kaggle Competition, Toxic Comment Classification Challenge, 
#that train dataset contains 159571 rows and 8 columns, which are id, comment_text, 
#toxic, sever_toxic, obscene, threat, insult and identity_hate.
#The test dataset has over 150000 records.
df_train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv') 
df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
train_input = df_train['comment_text']
test_input = df_test['comment_text']
# Define a function to read the FastText Pre-trained Word Embedding in to a dictionary.
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'))
del embeddings_index['2000000'] # The first row of the file is useless, so delete it.
len(embeddings_index) 
#FastText Word Embedding file contains 2500000 words including punctuations.
#It doesn't contains 0-9 and words like I'm, can't and etc.
max_features = 100000
maxlen = 170 
#Set the max length of each comment. If it is longer than 150 then cut if off,
#if it is shorter than 150 then pad it up to 150.
#This max length can be choosen in different ways. 
#Here it is a number that near 80 percentile of all comment length in training dataset.
# Define data cleaning function
def clean(string):
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub("[^A-Za-z\(\)\,\.\?\'\!]", " ", string)
    string = re.sub("\'m", ' am ', string)
    string = re.sub("\'s", ' is ', string)
    string = re.sub("can\'t", 'cannot ', string)
    string = re.sub("n\'t", ' not ', string)
    string = re.sub("\'ve", ' have ', string)
    string = re.sub("\'re", ' are ', string)
    string = re.sub("\'d", " would ", string)
    string = re.sub("\'ll", " will ", string)
    string = re.sub("\,", " , ", string)
    string = re.sub("\'", " ' ", string)
    string = re.sub("\.", " . ", string)
    string = re.sub("\!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r'\s{2,}', ' ', string.lower())
    return string
x_train = train_input.apply(clean)
y_train = df_train[['toxic','severe_toxic',"obscene", "threat", "insult", "identity_hate"]]
x_test = test_input.apply(clean)
#After data clean there might be some record have nothing in comment_text, fill with a word.
x_train = x_train.fillna('fillna')
x_test = x_test.fillna('fillna')
#Create the dictionary whose keys contains all words in train dataset that also shown 
#in FastText word embeddings.
lst = []
for line in x_train:
    lst += line.split()
    
count = Counter(lst)
for k in list(count.keys()):
    if k not in embeddings_index:
        del count[k]
len(count)
count = dict(sorted(count.items(), key=lambda x: -x[1]))
count = {k:v for (k,v) in count.items() if v >= 2}
len(count)
count = dict(zip(list(count.keys()),range(1,64349 + 1)))
embedding_matrix = {}
for key in count:
    embedding_matrix[key] = embeddings_index[key]
#Create teh word embedding matrix where the first element is all zeros which is for word
#that is not shown and the padding elements.
W = np.zeros((1,300))
W = np.append(W, np.array(list(embedding_matrix.values())),axis=0)
W = W.astype(np.float32, copy=False)
W.shape
#Same Step for text dataset.
lst = []
for line in x_test:
    lst += line.split()
    
count_test = Counter(lst)
for k in list(count_test.keys()):
    if k not in embedding_matrix:
        del count_test[k]
    else:
        count_test[k] = count[k]
len(count_test)
#Release memory.
del lst
gc.collect()
#Make the train dataset to be a sequence of ids of words.
for i in range(len(x_train)):
    temp = x_train[i].split()
    for word in temp[:]:
        if word not in count:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = count[temp[j]]
    x_train[i] = temp
for i in range(len(x_test)):
    temp = x_test[i].split()
    for word in temp[:]:
        if word not in count_test:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = count_test[temp[j]]
    x_test[i] = temp
#Create evaluation dataset.
#Xtrain, Xval, ytrain, yval = train_test_split(x_train, y_train, train_size=0.96, random_state=123)
#Pad sequence to 170 length.
train_x = sequence.pad_sequences(list(x_train), maxlen = maxlen)
test_x = sequence.pad_sequences(list(x_test), maxlen = maxlen)
del embeddings_index
gc.collect()
filter_sizes = [1,2,3,4,5]
num_filters = 32
batch_size = 256
#This large batch_size is specially for this case. Usually it is between 64-128.
num_filters_total = num_filters * len(filter_sizes)
embedding_size = 300
sequence_length = 170
num_epochs = 3 #Depends on your choice.
dropout_keep_prob = 0.9
input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
input_y = tf.placeholder(tf.float32, [None,6], name = "input_y") 
embedded_chars = tf.nn.embedding_lookup(W, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
def CNN(data):
    pooled_outputs = []
    
    for i, filter_size in enumerate(filter_sizes):
        
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        
        w = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.05), name = "w")
        b = tf.Variable(tf.truncated_normal([num_filters], stddev = 0.05), name = "b")
            
        conv = tf.nn.conv2d(
            data,
            w,
            strides = [1,1,1,1],
            padding = "VALID",
            name = "conv"
        )
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
        pooled = tf.nn.max_pool(
            h,
            ksize = [1, sequence_length - filter_size + 1, 1, 1],
            strides = [1,1,1,1],
            padding = "VALID",
            name = "pool"
        )
        
        pooled_outputs.append(pooled)
    
    #return pooled_outputs
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat
h_pool_flat = CNN(embedded_chars_expanded)
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
#In the first dense layer, reduce the node to half.
wd1 = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total/2)], stddev=0.05), name = "wd1")
bd1 = tf.Variable(tf.truncated_normal([int(num_filters_total/2)], stddev = 0.05), name = "bd1")
layer1 = tf.nn.xw_plus_b(h_drop, wd1, bd1, name = 'layer1') # Do wd1*h_drop + bd1
layer1 = tf.nn.relu(layer1)
#Second dense layer, reduce the outputs to 6.
wd2 = tf.Variable(tf.truncated_normal([int(num_filters_total/2),6], stddev = 0.05), name = 'wd2')
bd2 = tf.Variable(tf.truncated_normal([6], stddev = 0.05), name = "bd2")
layer2 = tf.nn.xw_plus_b(layer1, wd2, bd2, name = 'layer2') 
prediction = tf.nn.sigmoid(layer2)# Make it to be 0-1.
#pred_clipped = tf.clip_by_value(prediction, 1e-10, 0.9999999) 
#For some special loss function clip is necessary. Like log(x).
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = layer2, labels = input_y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0007).minimize(loss)
#when learning rate set to 0.0007, the mean of threat is not 0, but when it is 0.001, it becomes 0 again.
#Learning rates usually is small for CNN compared with pure neural network. 
#Need to define a approriate learning rate before you run on the whole dataset.
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction), input_y), tf.float32))
#correct_prediction = tf.equal(tf.argmax(input_y, 1), tf.argmax(prediction, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Define batch generation function.
def generate_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    l = 0
    for epoch in range(num_epochs):
        l += 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
#For Test data. Can use generate_batch function.
def blocks(data, block_size):
    data = np.array(data)
    data_size = len(data)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        if block_num == 0:
            print("prediction start!")
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        yield data[start_index:end_index]
# The reason to create 7 different batches here is because 
#I want to make the data totally shuffled to reduce the risk that one batch have all 0.
batch1 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch2 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch3 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch4 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch5 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch6 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch7 = generate_batch(list(zip(np.array(train_x), y_train['toxic'], y_train['severe_toxic'], y_train['obscene'], y_train['threat'], y_train['insult'], y_train['identity_hate'])), batch_size, 1)
batch_bag = [batch1,batch2,batch3,batch4,batch5,batch6,batch7]
test_blocks = blocks(list(np.array(test_x)), 1000)
train_x.shape
int((159571-1)/256)+1
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init_op)
    i = 0
    for batches in batch_bag:
        i += 1
        print('Epoch: ' + str(i) + ' start!')
        avg_acc = 0
        avg_loss = 0
        for batch in batches:
            batch = pd.DataFrame(batch, columns = ['a','b','c','d','e','g','f'])
            x_batch = pd.DataFrame(list(batch['a']))
            y_batch = batch.loc[:, batch.columns != 'a']
            _,c, acc = sess.run([optimizer, loss, accuracy],feed_dict = {input_x: x_batch, input_y: y_batch})
            avg_loss += c
            avg_acc += acc
        avg_loss = avg_loss/624
        avg_acc = avg_acc/624
        print('Epoch:' + str(i) + ' loss is ' + str(avg_loss) + ', train accuracy is ' + str(avg_acc))
        #print('Evaluation Accuracy: ')
        #print(accuracy.eval({input_x: val_x, input_y: yval}))
    
    print('Training Finish!')
    
    df = pd.DataFrame()
    for block in test_blocks:
        block = pd.DataFrame(block)
        pred = sess.run(prediction, feed_dict = {input_x: block})
        df = df.append(pd.DataFrame(pred))
    
    print('Prediction Finish!')
 
df.round().mean()
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = np.array(df)
submission.to_csv('submission.csv', index=False)