%matplotlib inline



# Imports

import numpy as np

import pandas as pd

import tensorflow as tf

from matplotlib import pylab

from bs4 import BeautifulSoup

from functools import reduce

from sklearn.manifold import TSNE

from IPython.display import display



# Convert csv files into dataframes

biology_pd = pd.read_csv('../input/biology.csv')

cooking_pd = pd.read_csv('../input/cooking.csv')

cryptology_pd = pd.read_csv('../input/crypto.csv')

diy_pd = pd.read_csv('../input/diy.csv')

robotics_pd = pd.read_csv('../input/robotics.csv')

travel_pd = pd.read_csv('../input/travel.csv')

test_pd = pd.read_csv('../input/test.csv')



# Print dataframe heads

print('Biology: %i questions' % biology_pd.shape[0])

display(biology_pd.head())

print('Cooking: %i questions' % cooking_pd.shape[0])

display(cooking_pd.head())

print('Crytology: %i questions' % cryptology_pd.shape[0])

display(cryptology_pd.head())

print('DIY: %i questions' % diy_pd.shape[0])

display(diy_pd.head())

print('Robotics: %i questions' % robotics_pd.shape[0])

display(robotics_pd.head())

print('Travel: %i questions' % travel_pd.shape[0])

display(travel_pd.head())

print('Test: %i questions' % test_pd.shape[0])

display(test_pd.head())
# Stop words from Stanford's NLP codebase: 

# github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface



stop_words = ["", " ", "a", "about", "above", "after", "again", "against", "all", "am", "an", 

              "and","any", "are", "aren't", "at", "be", "because", "been", "before", "being", 

              "below", "between", "both", "but", "by", "can", "can't",  "cannot", "could",

              "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", 

              "during", "each", "few", "for", "from", "further", "had",  "hadn't","has", 

              "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 

              "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 

              "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", 

              "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", 

              "myself", "no", "nor", "not" , "of", "off", "on", "once", "only", "or", "other",

              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", 

              "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 

              "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 

              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 

              "they've", "this", "those", "to", "too", "under", "until", "up", "very", "was",

              "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", 

              "what's", "when", "when's", "where", "where's", "which", "while", "who", 

              "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't","you", 

              "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", 

              "yourselves", "return", "arent", "cant", "couldnt", "didnt", "doesnt", "dont", 

              "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets",

              "mustnt", "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", 

              "theyve", "wasnt", "were", "werent", "whats", "whens", "wheres", "whos", "whys",

              "wont", "wouldnt", "youd", "youll", "youre", "youve"]
# Convert dataframes to ndarrays

biology_np = biology_pd[['title', 'content', 'tags']].as_matrix()

cooking_np = cooking_pd[['title', 'content', 'tags']].as_matrix()

cryptology_np = cryptology_pd[['title', 'content', 'tags']].as_matrix()

diy_np = diy_pd[['title', 'content', 'tags']].as_matrix()

robotics_np = robotics_pd[['title', 'content', 'tags']].as_matrix()

travel_np = travel_pd[['title', 'content', 'tags']].as_matrix()

test_np = test_pd[['title', 'content']].as_matrix()



# Parse html

def parse_html(data_np):    

    for i in range(data_np.shape[0]):

        soup = BeautifulSoup(data_np[i,1], 'html.parser')

        soup = soup.get_text()

        soup = BeautifulSoup(soup, 'html.parser')

        soup = soup.decode('utf8')

        soup = soup.replace('\n', ' ')

        data_np[i,1] = soup



parse_html(biology_np)

parse_html(cooking_np)

parse_html(cryptology_np)

parse_html(diy_np)

parse_html(robotics_np)

parse_html(travel_np)

parse_html(test_np)



# Create datasets and labels

def to_list(data):    

    for i in range(len(data)):

        data[i] = [''.join([ch for ch in s if ch.isalnum()]) for s in data[i].split(' ')]

        #data[i] = [x for x in data[i] if len(x) > 0]

    #return [x for xs in data for x in xs if len(x) > 0]

    return [x for xs in data for x in xs if x not in stop_words]



biology_x = to_list(biology_np[:,0] + ' ' + biology_np[:,1])

biology_y = to_list(biology_np[:,2])

cooking_x = to_list(cooking_np[:,0] + ' ' + cooking_np[:,1])

cooking_y = to_list(cooking_np[:,2])

cryptology_x = to_list(cryptology_np[:,0] + ' ' + cryptology_np[:,1])

cryptology_y = to_list(cryptology_np[:,2])

diy_x = to_list(diy_np[:,0] + ' ' + diy_np[:,1])

diy_y = to_list(diy_np[:,2])

robotics_x = to_list(robotics_np[:,0] + ' ' + robotics_np[:,1])

robotics_y = to_list(robotics_np[:,2])

travel_x = to_list(travel_np[:,0] + ' ' + travel_np[:,1])

travel_y = to_list(travel_np[:,2])

test_x = to_list(test_np[:,0] + ' ' + test_np[:,1])



# Print sample data and labels

print('Biology data: %i words' % len(biology_x))

print(biology_x[:50])

print('\nBiology labels: %i words' % len(biology_y))

print(biology_y[:10])

print('\nCooking data: %i words' % len(cooking_x))

print(cooking_x[:50])

print('\nCooking labels: %i words' % len(cooking_y))

print(cooking_y[:10])

print('\nCryptology data: %i words' % len(cryptology_x))

print(cryptology_x[:50])

print('\nCryptology labels: %i words' % len(cryptology_y))

print(cryptology_y[:10])

print('\nDiy data: %i words' % len(diy_x))

print(diy_x[:50])

print('\nDiy labels: %i words' % len(diy_y))

print(diy_y[:10])

print('\nRobotics data: %i words' % len(robotics_x))

print(robotics_x[:50])

print('\nRobotics labels: %i words' % len(robotics_y))

print(robotics_y[:10])

print('\nTravel data: %i words' % len(travel_x))

print(travel_x[:50])

print('\nTravel labels: %i words' % len(travel_y))

print(travel_y[:10])

print('\nTest data: %i words' % len(test_x))

print(test_x[:50])
batch_size = 64

embedding_size = 64

vocab_size = 10000

num_sampled = 64

num_context = 4

data_index = 0



def create_batch(data, data_index, num_context):

    batch_targets = np.ndarray([batch_size], np.int32)

    batch_contexts = np.ndarray([batch_size, 1], np.int32)

    

    for i in range(0, batch_size, num_context): 

        context_indexes = [x for x in range(data_index, data_index + num_context + 1)]

        del context_indexes[len(context_indexes) // 2]

        batch_targets[i:i + num_context] = data_index + num_context // 2

        batch_contexts[i:i + num_context, 0] = context_indexes

        data_index = (data_index + 1) % len(data)



    return batch_targets, batch_contexts



test_batch_targets, test_batch_contexts = create_batch(robotics_x,

                                                       data_index,

                                                       num_context)



print('Original: ' + str(test_x[:batch_size // num_context + num_context]) + '\n')

print('Target: ' + str([test_x[x] for x in test_batch_targets]) + '\n')

print('Context: ' + str([test_x[x[0]] for x in test_batch_contexts]))
graph = tf.Graph()

with graph.as_default():

    train_x = tf.placeholder(tf.int32, [batch_size])

    train_y = tf.placeholder(tf.int32, [batch_size, 1])

        

    embedding_space = tf.Variable(tf.random_uniform([vocab_size, embedding_size]))

    embedded_train_x = tf.nn.embedding_lookup(embedding_space, train_x)

    weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size]))

    biases = tf.Variable(tf.zeros([vocab_size]))

    

    loss = tf.reduce_mean(tf.nn.nce_loss(weights, 

                                         biases, 

                                         embedded_train_x, 

                                         train_y, 

                                         num_sampled, 

                                         vocab_size))



    optimizer = tf.train.AdamOptimizer().minimize(loss)   

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_space), 1, keep_dims=True))

    normalized_embedding_space = embedding_space / norm    
num_steps = 10001

data_index = 0



with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()   



    for step in range(num_steps):

        batch_x, batch_y = create_batch(robotics_x, data_index, num_context) 

        _, l = session.run([optimizer, loss], {train_x:batch_x, train_y:batch_y})

        

        if step % 1000 == 0:            

            print('Loss at step %i: %.2f' % (step, l))

            

    final_embedding_space = normalized_embedding_space.eval()
tsne = TSNE()

tsne_embedding_space = tsne.fit_transform(final_embedding_space[:100])



pylab.figure(figsize=(9, 9))



for i in range(len(tsne_embedding_space)):

    x, y = tsne_embedding_space[i, :]

    pylab.scatter(x, y)

    pylab.annotate(robotics_x[i], xy=(x, y))



pylab.show()