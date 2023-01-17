import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

import pandas as pd

import string

import nltk

import random

import collections

from nltk.corpus import stopwords
batch_size = 50

embedding_size = 200

vocabulary_size = 10000

generations = 1000

print_loss_every = 500

num_sampled = int(batch_size/2)

window_size = 2

print_valid_every = 200
nltk.download('stopwords')

stops = stopwords.words('english')
valid_words = ['android', 'love', 'hate', 'silly', 'want']
data = pd.read_csv("/kaggle/input/my-data/googleplaystore_user_reviews.csv")

data.head()
data_reviews = pd.DataFrame()

data_reviews["Translated_Review"] =  data["Translated_Review"]

print("# Registers:"+str(data_reviews.shape[0]))

for column in data_reviews.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data_reviews[column]).values.ravel().sum()))
#Remove the na´s

corpus=data_reviews.dropna()

corpus.head()
def normalize_text(texts, stops):

    texts = [x.lower() for x in texts]

    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    texts = [' '.join(word for word in x.split() if word not in (stops)) for x in texts]

    texts = [' '.join(x.split()) for x in texts]

    texts.remove('')

    return texts
texts = normalize_text(np.asarray(corpus['Translated_Review']), stops)
def build_dictionary(sentences, vocabulary_size):

    split_sentences = [s.split() for s in sentences] # obtain array of the words for each sentence

    words = [x for sublist in split_sentences for x in sublist] # get each word in the sentences

    # Get the 10000 = vocabulary_size most common words

    count = [['RARE', -1]] # if a word is not very common we assigned to RARE

    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    # Create dictionary each word will have a unique ID

    word_dict = {}

    for word, word_count in count:

        word_dict[word] = len(word_dict)

    return word_dict
word_dict = build_dictionary(texts, vocabulary_size)

#Reverse the dictionary

word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))

print("Word_dict_reverse id 1="+word_dict_rev[1]+" "+"Word_dict game="+ str(word_dict['game']))
def text_to_numbers(sentences, word_dict):

    data = []

    counter = 0 

    for sentence in sentences:

        sentence_data = []

        counter = counter + 1

        for word in sentence:

            if word in word_dict:

                word_ix = word_dict[word] # position/ID for word in the word dict

            else:

                word_ix = 0 # position/ID for word RARE

            sentence_data.append(word_ix)

        data.append(sentence_data)

    return data
#Encode each sentence

text_data = text_to_numbers(texts, word_dict)

# Remove empty arrays

text_data=[text for text in text_data if text != []]
def generate_batch_data(sentences, batch_size, window_size, method = 'skip_gram'):

    '''

        Skip Gram example for 'come' (input,label)

        Mi perro come su comida -> (Mi, come), (perro, come), (su, come), (comida, come)

    '''

    batch_data = []

    label_data = []

    

    while len(batch_data) < batch_size:

        

        rand_sentences = np.random.choice(sentences) 

        

        if rand_sentences == []:

            print("Empty")

        # X is the encoded word

        # ix is position in the array

        # window_seq are the combinations of neighbors for each word and includes the word

        window_seq = [rand_sentences[max((ix-window_size),0):(ix+window_size+1)] 

                      for ix, x in enumerate(rand_sentences)] 

        

        #label_idx is the label position

        label_idx = [ix if ix < window_size else window_size for ix, x in enumerate(window_seq)]

        

        if method == 'skip_gram':

            #x[y] is the central word

            batch_and_labels = [(x[y], x[:y]+x[(y+1):]) for x,y in zip(window_seq, label_idx)]

            tuple_data = [(x,y_) for x, y in batch_and_labels for y_ in y]

        else:

            raise ValueError("Método {} no implementado".format(method))

        

        try:

            batch, labels = [list(x) for x in zip(*tuple_data)]

            batch_data.extend(batch[:batch_size])

            label_data.extend(labels[:batch_size])

        except:

            continue

       

    batch_data = batch_data[:batch_size]

    label_data = label_data[:batch_size]

    

    batch_data = np.array(batch_data)

    label_data = np.transpose(np.array([label_data]))

    

    return (batch_data, label_data)
# Inputs, target, valid_dataset

valid_examples = [word_dict[x] for x in valid_words] 

x_inputs = tf.placeholder(tf.int32, shape =[batch_size])

y_target = tf.placeholder(tf.int32, shape = [batch_size,1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
# Hidden Layer,embbedding_size = neurons

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1, 1))

embed = tf.nn.embedding_lookup(embeddings, x_inputs) # Lookup table for the coordinates
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0/np.sqrt(embedding_size)))

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,

                                    inputs=embed, labels=y_target, 

                                     num_sampled = num_sampled, num_classes=vocabulary_size))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keepdims=True))

normalized_embeddings = embeddings/norm

valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
session = tf.Session()

init = tf.global_variables_initializer()

session.run(init)
loss_vect = []

loss_x_vect = []

for i in range(generations):

    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)

    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    session.run(optimizer, feed_dict=feed_dict)

    

    # Print loss 

    if (i+1) % print_loss_every == 0:

        loss_val = session.run(loss, feed_dict=feed_dict)

        loss_vect.append(loss_val)

        loss_x_vect.append(i+1)

        #print("batch_inputs:{}".format(batch_inputs))

        #print("Iteración {}, Pérdida: ".format(i+1, loss_val))

    

    ## Validate the 5 words selected 

    if (i+1) % print_valid_every == 0:

        sim = session.run(similarity, feed_dict=feed_dict)

        for j in range(len(valid_words)):

            valid_word = word_dict_rev[valid_examples[j]]

            top_k = 10

            nearest = (-sim[j,:]).argsort()[1:top_k+1]

            log_string = "Palabras cercanas a {}:".format(valid_word)

            for k in range(top_k):

                close_word = word_dict_rev[nearest[k]]

                log_string = "%s %s, "%(log_string, close_word)

            #print(log_string)
app_reviews = data

app_reviews = app_reviews.dropna()

app_reviews.head()
app_reviews.head()
def normalize_sentence(sentence):

    stops = stopwords.words('english')

    sentence = normalize_text([sentence,''], stops)

    return sentence[0]

    
app_reviews['Translated_Review'] = [normalize_sentence(sentence) for sentence in app_reviews['Translated_Review']]

app_reviews['Sentiment'] = [ 1 if sentiment == 'Positive' else 0 for sentiment in app_reviews['Sentiment']]
del app_reviews['Sentiment_Polarity']

del app_reviews['Sentiment_Subjectivity']

del app_reviews['App']

app_reviews.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(app_reviews['Translated_Review'],app_reviews['Sentiment'],test_size = 0.3, random_state=0)
max_words = 100

text_data_train = np.array(text_to_numbers(X_train, word_dict))

text_data_test = np.array(text_to_numbers(X_test, word_dict))

# Make that all sentences are the lenght of max_words

text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])

text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])
max_words = 100

# Create Input and target 

x_data = tf.placeholder(shape = [None, max_words], dtype = tf.int32)

y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)
# Define hidden layers

A = tf.Variable(tf.random_normal(shape = [embedding_size, 1]))

b = tf.Variable(tf.random_normal(shape = [1,1]))



# Embbeding layer

embeddings # The previous embeddings 

embed = tf.nn.embedding_lookup(embeddings, x_data)

embed_avg = tf.reduce_mean(embed,1)



#Output

model_output = tf.add(tf.matmul(embed_avg, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.round(tf.sigmoid(model_output))

predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)

accuracy = tf.reduce_mean(predictions_correct)
my_optim = tf.train.AdagradOptimizer(0.005)

train_step = my_optim.minimize(loss)
session = tf.Session()

init = tf.global_variables_initializer()

session.run(init)
train_loss = []

test_loss = []

train_acc = []

test_acc = []

i_data = []



for i in range(10000):

    rand_idx = np.random.choice(len(text_data_train), size = batch_size)

    

    rand_x = np.array(text_data_train)[rand_idx]

    rand_x = np.asarray(rand_x)

   

    Y_train_array = Y_train.to_numpy()

    rand_y = Y_train_array[rand_idx].tolist()

    rand_y = [[x]for x in rand_y]

    

    session.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    

    if(i+1)%100==0:

        i_data.append(i+1)

        

        #TRAIN LOSS

        train_loss_temp = session.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})

        train_loss.append(train_loss_temp)

        

        # Prepare Test Data for testing the NN

        X_test_array = np.array(text_data_test)

        X_test_array = np.asarray(X_test_array) 

        

        Y_test_array = Y_test.to_numpy()

        Y_test_array = Y_test_array.tolist()

        Y_test_array = [[x]for x in Y_test_array]

        

        ## TEST LOSS

        test_loss_temp = session.run(loss, feed_dict={x_data: X_test_array, y_target: Y_test_array})

        test_loss.append(test_loss_temp)

        

        ## TRAINING ACC

        train_acc_temp = session.run(accuracy, feed_dict={x_data: rand_x, y_target:rand_y})

        train_acc.append(train_acc_temp)

        

        ## Testing ACC

        test_acc_temp = session.run(accuracy, feed_dict={x_data: X_test_array, y_target: Y_test_array})

        test_acc.append(test_acc_temp)

   

    if(i+1)%500==0:

        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]

        acc_and_loss = [np.round(x,3) for x in acc_and_loss]

        print("Paso #{}, Train Loss {}, Test Loss {}. Train Acc {}, Test Acc{}".format(*acc_and_loss))

    
import pandas as pd

googleplaystore_user_reviews = pd.read_csv("../input/my-data/googleplaystore_user_reviews.csv")