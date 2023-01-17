import numpy as np

import pandas as pd
!ls ../input
import codecs

input_file = codecs.open('../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv', 

                         'r',

                         encoding='utf-8', 

                         errors='replace')

output_file = open('clean_socialmedia-disaster.csv', 'w')

    

for line in input_file:

    out = line

    output_file.write(line)

input_file.close()
output_file.close()
df = pd.read_csv('clean_socialmedia-disaster.csv')
df.head()
df.shape
df.choose_one.unique()
df['choose_one'].value_counts().plot(kind='bar')
df = df[df.choose_one != "Can't Decide"]
df.shape
df = df[['text','choose_one']]
df.head()
df['relevant'] = df.choose_one.map({'Relevant':1,'Not Relevant':0})
df.head()
df.describe()
import spacy
nlp = spacy.load('en',disable=['tagger','parser','ner'])
from tqdm import tqdm, tqdm_notebook

tqdm.pandas(tqdm_notebook)
df['lemmas'] = df["text"].progress_apply(lambda row: 

                                         [w.lemma_ for w in nlp(row)])
df.head()
df['joint_lemmas'] = df['lemmas'].progress_apply(lambda row: ' '.join(row))
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['joint_lemmas'], 

                                                    df['relevant'], 

                                                    test_size=0.2,

                                                    random_state=40)


from sklearn.feature_extraction.text import CountVectorizer



count_vectorizer = CountVectorizer(max_features=5000)

X_train_counts = count_vectorizer.fit_transform(X_train)

X_test_counts = count_vectorizer.transform(X_test)
X_train_counts.shape
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib



 



lsa = TruncatedSVD(n_components=2)

lsa.fit(X_train_counts)

lsa_scores = lsa.transform(X_train_counts)





fig = plt.figure(figsize=(16, 16))   

colors = ['orange','blue']



plt.scatter(lsa_scores[:,0], 

            lsa_scores[:,1], 

            s=8, alpha=.8, 

            c=y_train,

            cmap=matplotlib.colors.ListedColormap(colors))



ir_patch = mpatches.Patch(color='Orange',label='Irrelevant')



dis_patch = mpatches.Patch(color='Blue',label='Disaster')



plt.legend(handles=[ir_patch, dis_patch], prop={'size': 30})



plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

clf = LogisticRegression()



clf.fit(X_train_counts, y_train)



y_predicted = clf.predict(X_test_counts)
accuracy_score(y_test, y_predicted)
import numpy as np

import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.winter):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, fontsize=20)

    plt.yticks(tick_marks, classes, fontsize=20)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.



    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 

                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    

    plt.tight_layout()

    plt.ylabel('True label', fontsize=30)

    plt.xlabel('Predicted label', fontsize=30)



    return plt
cm = confusion_matrix(y_test, y_predicted)

fig = plt.figure(figsize=(10, 10))

plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster'], normalize=False, title='Confusion matrix')

plt.show()

print(cm)

def get_most_important_features(vectorizer, model, n=5):

    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}

    

    # loop for each class

    classes ={}

    for class_index in range(model.coef_.shape[0]):

        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]

        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)

        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])

        bottom = sorted_coeff[-n:]

        classes[class_index] = {

            'tops':tops,

            'bottom':bottom

        }

    return classes



importance = get_most_important_features(count_vectorizer, clf, 10)
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):

    y_pos = np.arange(len(top_words))

    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]

    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    

    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]

    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    

    top_words = [a[0] for a in top_pairs]

    top_scores = [a[1] for a in top_pairs]

    

    bottom_words = [a[0] for a in bottom_pairs]

    bottom_scores = [a[1] for a in bottom_pairs]

    

    fig = plt.figure(figsize=(10, 10))  



    plt.subplot(121)

    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)

    plt.title('Irrelevant', fontsize=20)

    plt.yticks(y_pos, bottom_words, fontsize=14)

    plt.suptitle('Key words', fontsize=16)

    plt.xlabel('Importance', fontsize=20)

    

    plt.subplot(122)

    plt.barh(y_pos,top_scores, align='center', alpha=0.5)

    plt.title('Disaster', fontsize=20)

    plt.yticks(y_pos, top_words, fontsize=14)

    plt.suptitle(name, fontsize=16)

    plt.xlabel('Importance', fontsize=20)

    

    plt.subplots_adjust(wspace=0.8)

    plt.show()



top_scores = [a[0] for a in importance[0]['tops']]

top_words = [a[1] for a in importance[0]['tops']]

bottom_scores = [a[0] for a in importance[0]['bottom']]

bottom_words = [a[1] for a in importance[0]['bottom']]



plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=10000)



X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)
lsa = TruncatedSVD(n_components=2)

lsa.fit(X_train_tfidf)

lsa_scores = lsa.transform(X_train_tfidf)



fig = plt.figure(figsize=(16, 16))          

colors = ['orange','blue']



plt.scatter(lsa_scores[:,0], 

            lsa_scores[:,1], 

            s=8, alpha=.8, 

            c=y_train,

            cmap=matplotlib.colors.ListedColormap(colors))



ir_patch = mpatches.Patch(color='Orange',label='Irrelevant')



dis_patch = mpatches.Patch(color='Blue',label='Disaster')



plt.legend(handles=[ir_patch, dis_patch], prop={'size': 30})

plt.show()
clf_tfidf = LogisticRegression()

clf_tfidf.fit(X_train_tfidf, y_train)



y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
accuracy_score(y_pred=y_predicted_tfidf, y_true=y_test)
cm2 = confusion_matrix(y_test, y_predicted_tfidf)

fig = plt.figure(figsize=(10, 10))

plot = plot_confusion_matrix(cm2, classes=['Irrelevant','Disaster'], normalize=False, title='Confusion matrix')

plt.show()

print("TFIDF confusion matrix")

print(cm2)

print("BoW confusion matrix")

print(cm)
importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
top_scores = [a[0] for a in importance_tfidf[0]['tops']]

top_words = [a[1] for a in importance_tfidf[0]['tops']]

bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]

bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]



plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
from keras.preprocessing.text import Tokenizer

import numpy as np



max_words = 10000 # We will only consider the 10K most used words in this dataset
tokenizer = Tokenizer(num_words=max_words) # Setup

tokenizer.fit_on_texts(df['joint_lemmas']) # Generate tokens by counting frequency

sequences = tokenizer.texts_to_sequences(df['joint_lemmas']) # Turn text into sequence of numbers
word_index = tokenizer.word_index

print('Token for "the"',word_index['the'])

print('Token for "Movie"',word_index['movie'])
from keras.preprocessing.sequence import pad_sequences

maxlen = 140 # Make all sequences 140 words long

data = pad_sequences(sequences, maxlen=maxlen)

print(data.shape) # We have 25K, 140 word sequences now
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,

                                                    df['relevant'],

                                                    test_size = 0.2, 

                                                    shuffle=True, 

                                                    random_state = 42)
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense



embedding_dim = 50



model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

model.add(Flatten())

#model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
type(data)
history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense



embedding_dim = 50



model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

model.add(LSTM(32))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
y_predicted_nn = model.predict_classes(X_test)

accuracy_score(y_pred=y_predicted_nn, y_true=y_test)
!ls ../input/glove6b50d
import os

glove_dir = '../input/glove6b50d' # This is the folder with the dataset



embeddings_index = {} # We create a dictionary of word -> embedding

f = open(os.path.join(glove_dir, 'glove.6B.50d.txt')) # Open file



# In the dataset, each line represents a new word embedding

# The line starts with the word and the embedding values follow

for line in f:

    values = line.split()

    word = values[0] # The first value is the word, the rest are the values of the embedding

    embedding = np.asarray(values[1:], dtype='float32') # Load embedding

    embeddings_index[word] = embedding # Add embedding to our embedding dictionary

f.close()



print('Found %s word vectors.' % len(embeddings_index))
all_embs = np.stack(embeddings_index.values())

emb_mean = all_embs.mean() # Calculate mean

emb_std = all_embs.std() # Calculate standard deviation

emb_mean,emb_std
embedding_dim = 50



word_index = tokenizer.word_index

nb_words = min(max_words, len(word_index)) # How many words are there actually



# Create a random matrix with the same mean and std as the embeddings

embedding_matrix = np.random.normal(emb_mean, 

                                    emb_std, 

                                    (nb_words, embedding_dim))



# The vectors need to be in the same position as their index. 

# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on



# Loop over all words in the word index

for word, i in word_index.items():

    # If we are above the amount of words we want to use we do nothing

    if i >= max_words: 

        continue

    # Get the embedding vector for the word

    embedding_vector = embeddings_index.get(word)

    # If there is an embedding vector, put it in the embedding matrix

    if embedding_vector is not None: 

        embedding_matrix[i] = embedding_vector
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense

model = Sequential()

model.add(Embedding(max_words, 

                    embedding_dim, 

                    input_length=maxlen, 

                    weights = [embedding_matrix], trainable = False))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
y_predicted_nn = model.predict_classes(X_test)

accuracy_score(y_pred=y_predicted_nn, y_true=y_test)
from keras.layers import CuDNNLSTM

model = Sequential()

model.add(Embedding(max_words, 

                    embedding_dim, 

                    input_length=maxlen, 

                    weights = [embedding_matrix], trainable = False))

model.add(CuDNNLSTM(32))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
y_predicted_nn = model.predict_classes(X_test)

accuracy_score(y_pred=y_predicted_nn, y_true=y_test)
from keras.layers import Bidirectional

model = Sequential()

model.add(Embedding(max_words, 

                    embedding_dim, 

                    input_length=maxlen, 

                    weights = [embedding_matrix], trainable = False))

model.add(Bidirectional(CuDNNLSTM(32)))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
from keras.layers import Bidirectional

model = Sequential()

model.add(Embedding(max_words, 

                    embedding_dim, 

                    input_length=maxlen, 

                    weights = [embedding_matrix], trainable = False))

model.add(Bidirectional(CuDNNLSTM(64,return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(64,return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(64,return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(32)))



model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
from keras.layers import Multiply, CuDNNLSTM, Permute, Reshape, Dense, Lambda, Input, Embedding, RepeatVector

import keras.backend as K

from keras.layers import LSTM

from keras.models import Model
INPUT_DIM = embedding_dim

TIME_STEPS = maxlen

SINGLE_ATTENTION_VECTOR = False
from keras.layers import *

from keras.layers.core import *

from keras.layers.recurrent import LSTM

from keras.models import *
def attention_3d_block(inputs,time_steps,single_attention_vector = False):

    # inputs.shape = (batch_size, time_steps, input_dim)

    input_dim = int(inputs.shape[2])

    a = Permute((2, 1),name='Attent_Permute')(inputs)

    a = Reshape((input_dim, time_steps),name='Reshape')(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(time_steps, activation='softmax', name='Attent_Dense')(a) # Create attention vector

    if single_attention_vector:

        # If we just need one attention vector it over all input dimensions

        a = Lambda(lambda x: K.mean(x, axis=1), name='Dim_reduction')(a) 

        a = RepeatVector(input_dim, name='Repeat')(a)

    a_probs = Permute((2, 1), name='Attention_vec')(a) # Swap time steps, input dim axis back

    output_attention_mul = Multiply(name='Attention_mul')([inputs, a_probs]) # Multiply input with attention vector

    return output_attention_mul
input_tokens = Input(shape=(maxlen,),name='input')



embedding = Embedding(max_words, 

                      embedding_dim, 

                      input_length=maxlen, 

                      weights = [embedding_matrix], 

                      trainable = False, name='embedding')(input_tokens)



attention_mul = attention_3d_block(inputs = embedding,

                                   time_steps = maxlen,

                                   single_attention_vector = True)



lstm_out = CuDNNLSTM(32, return_sequences=True, name='lstm')(attention_mul)







attention_mul = Flatten(name='flatten')(attention_mul)

output = Dense(1, activation='sigmoid',name='output')(attention_mul)

model = Model(input_tokens, output)

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

history = model.fit(X_train, y_train,

                    epochs=10,

                    batch_size=32,

                    validation_data=(X_test, y_test))
nlp = spacy.load('en')
sup1 = nlp('I would like to open a new checking account')

sup2 = nlp('How do I open a checking account?')
sup1.similarity(sup2)
sup3 = nlp('I want to close my checking account')
sup1.similarity(sup3)
sup4 = nlp('I like checking the news')
sup1.similarity(sup4)
import sense2vec
def attention_3d_block(inputs,maxlen,single_attention_vector = False):

    # inputs.shape = (batch_size, time_steps, input_dim)

    input_dim = int(inputs.shape[2])

    time_steps = int(inputs.shape[1])

    #print(input_dim,time_steps)

    a = Permute((2, 1))(inputs) # Swap axis 1 & 2

    a = Reshape((input_dim, maxlen))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(maxlen, activation='softmax')(a) # Create dense layer to apply to input

    if single_attention_vector:

        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)

        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)

    output_attention_mul = Multiply(name='attention_mul')([inputs, a_probs])

    return output_attention_mul