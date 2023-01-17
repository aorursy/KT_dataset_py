# Library imports

import re

import numpy as np

import pandas as pd

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

import tensorflow as tf



# Reading the Data

data=pd.read_csv('../input/TechCrunch.csv',sep=',',error_bad_lines=False,encoding='ISO-8859-1')



# Cleaning the Data

data['title']=data['title'].map(lambda x:re.sub(r'[^\x00-\x7F]+',' ',x))

data['url']=data['url'].map(lambda x:re.sub(r'[^\x00-\x7F]+',' ',x))



# ANALYSIS 1

# Create a graph of all the related elements with the size of the node denoting the importance

# and edges denoting the relationship with others\



def combineProperNouns(a):

    y=0

    while y <= len(a)-2:

        if(a[y][0].isupper()==True and a[y+1][0].isupper()==True):

            a[y]=str(a[y]) + '+' + str(a[y+1])

            a[y+1:]=a[y+2:]

        else:

            y=y+1

    return(a)



def recreateDataWithCombinedProperNouns(data):

    tempData=[]

    for x in data.split('.'):

        tempPhrase=[]

        for y in x.split(','):

            z=y.split(' ')

            z=[a for a in z if len(a) > 0]

            tempPhrase.append(' '.join(combineProperNouns(z)))

        tempData.append(','.join(tempPhrase))

    data='.'.join(tempData)

    return(data)



def removeDotsFromAcronyms(data):

    counter=0

    while counter < len(data) -2:

        if(data[counter]=='.' and data[counter+2]=='.'):

            #print("######{}#####{}#######{}####".format(counter,data[counter-1:counter+3],data[counter+1]))

            data=data[:counter] + str(data[counter+1]) + ' ' + data[counter+3:]

            counter=counter+1

        elif(data[counter]=='.' and data[counter-1].isupper()==True):

            #print("####{}####".format(data[counter-1:counter+1]))

            data=data[:counter] + data[counter+1:]

        else:

            counter=counter+1

    return(data)



# Stemming and Lemmatizing the data

def stemAndLemmatize(data,columnNames):

    wordnet_lemmatizer = WordNetLemmatizer()

    porter_stemmer=PorterStemmer()

    for columnName in columnNames:

        data[columnName]=data[columnName].map(lambda x: ' '.join([porter_stemmer.stem(y) for y in x.split(' ')]))

        data[columnName]=data[columnName].map(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(y) for y in x.split(' ')]))

    return(data)







data['newTitle']=data['title'].map(lambda x:recreateDataWithCombinedProperNouns(x))

data=stemAndLemmatize(data,['title'])

data['newTitle']=data['newTitle'].map(lambda x: ' '.join([ y for y in x.split(' ') if nltk.pos_tag(y.split())[0][1] not in ['DT','IN','PDT','TO'] ]))

data['newTitle']=data['newTitle'].map(lambda x: ' '.join([ y for y in x.split(' ') if len(y) > 1]))

#strData=recreateDataWithCombinedProperNouns(strData)

# We need access to certain bigrams from the internet

# This will be another problem statement



# Noun verb relationship

# For each line

# a) Get the Noun or consecutive Nouns

# b) Get the adjective just before and after the NOUN and attach it to that Noun

# c) Get the Verb

# d) Get the adverb just before and after the Verb and attach it to that Verb



# We will run a word2vec using tensorflow and check the related words that will come out

tagList=['NNS','NNP','NNPS','VB','VBD','VBG']

data['newTitle']=data['newTitle'].map(lambda x: ' '.join([ y for y in x.split(' ') if nltk.pos_tag(y.split())[0][1] in tagList ]))

wordList=set([y  for x in data['newTitle'].values for y in x.split(' ')])

#words=[y  for x in data['newTitle'].values for y in x.split(' ')]

#words=set(words)

print("The number of words are {}".format(len(wordList)))

# Number of unique words

vocab_size=len(wordList)



word2int={}

int2word={}



for i,word in enumerate(wordList):

    word2int[word]=i

    int2word[word]=i



words=[]

WINDOW_SIZE=2

for sentence in data['newTitle'].values:

    newSentence=sentence.split(' ')

    for word_index,word in enumerate(newSentence):

        for nb_word in newSentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(newSentence)) + 1]: 

            if nb_word != word:

                words.append([word,nb_word])



def to_one_hot(data_point_index, vocab_size):

    temp = np.zeros(vocab_size)

    temp[data_point_index] = 1

    return temp



x_train = [] # input word

y_train = [] # output word

for data_word in words:

    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))

    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

# convert them to numpy arrays

x_train = np.asarray(x_train)

y_train = np.asarray(y_train)



# Tensorflow Model

x = tf.placeholder(tf.float32, shape=(None, vocab_size))

y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number

W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))

b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias

hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))

b2 = tf.Variable(tf.random_normal([vocab_size]))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))    
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init) #make sure you do this!

# define the loss function:

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

# define the training step:

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10



print("We will start training now")



# train for n_iter iterations

for _ in range(n_iters):

    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})

    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
# We will get the intermediate representation

vectors = sess.run(W1 + b1)



from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)

np.set_printoptions(suppress=True)

vectors = model.fit_transform(vectors)



from sklearn import preprocessing

normalizer = preprocessing.Normalizer()

vectors =  normalizer.fit_transform(vectors, 'l2')
wordList=list(set([y  for x in data['newTitle'].values for y in x.split(' ')]))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,5))

for word in wordList[0:100]:

    #print(word, vectors[word2int[word]][1])

    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

plt.show()



fig, ax = plt.subplots(figsize=(10,5))

for word in wordList[100:200]:

    #print(word, vectors[word2int[word]][1])

    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

plt.show()
