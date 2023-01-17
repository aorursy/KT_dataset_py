import os

import re



import numpy as np

import tensorflow as tf



np.random.seed(1)

tf.set_random_seed(2)



import pandas as pd

import keras

# from tqdm import tqdm

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import class_weight

from sklearn.metrics import f1_score, classification_report, log_loss



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten

from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping



print(os.listdir('../input'))
#training constants

MAX_SEQ_LEN = 25 #this is based on a quick analysis of the len of sequences train['text'].apply(lambda x : len(x.split(' '))).quantile(0.95)

DEFAULT_BATCH_SIZE = 128
data = pd.read_csv('../input/first-gop-debate-twitter-sentiment/Sentiment.csv')

# data = data[data['sentiment'] != 'Neutral']

train, test = train_test_split(data, random_state = 42, test_size=0.1)

print(train.shape)

print(test.shape)
# Mapping of common contractions, could probbaly be done better

CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",

                       "It's": 'It is', "Can't": 'Can not',

                      }
def clean_text(text, mapping):

    replace_white_space = ["\n"]

    for s in replace_white_space:

        text = text.replace(s, " ")

    replace_punctuation = ["’", "‘", "´", "`", "\'", r"\'"]

    for s in replace_punctuation:

        text = text.replace(s, "'")

    

    # Random note: removing the URL's slightly degraded performance, it's possible the model learned that certain URLs were positive/negative

    # And was able to extrapolate that to retweets. Could also explain why re-training the Embeddings improves performance.

    # remove twitter url's

#     text = re.sub(r"http[s]?://t.co/[A-Za-z0-9]*","TWITTERURL",text)

    mapped_string = []

    for t in text.split(" "):

        if t in mapping:

            mapped_string.append(mapping[t])

        elif t.lower() in mapping:

            mapped_string.append(mapping[t.lower()])

        else:

            mapped_string.append(t)

    return ' '.join(mapped_string)
# Get tweets from Data frame and convert to list of "texts" scrubbing based on clean_text function

# CONTRACTION_MAPPING is a map of common contractions(e.g don't => do not)

train_text_vec = [clean_text(text, CONTRACTION_MAPPING) for text in train['text'].values]

test_text_vec = [clean_text(text, CONTRACTION_MAPPING) for text in test['text'].values]





# tokenize the sentences

tokenizer = Tokenizer(lower=False)

tokenizer.fit_on_texts(train_text_vec)

train_text_vec = tokenizer.texts_to_sequences(train_text_vec)

test_text_vec = tokenizer.texts_to_sequences(test_text_vec)



# pad the sequences

train_text_vec = pad_sequences(train_text_vec, maxlen=MAX_SEQ_LEN)

test_text_vec = pad_sequences(test_text_vec, maxlen=MAX_SEQ_LEN)



print('Number of Tokens:', len(tokenizer.word_index))

print("Max Token Index:", train_text_vec.max(), "\n")



print('Sample Tweet Before Processing:', train["text"].values[0])

print('Sample Tweet After Processing:', tokenizer.sequences_to_texts([train_text_vec[0]]), '\n')



print('What the model will interpret:', train_text_vec[0].tolist())

# One Hot Encode Y values:

encoder = LabelEncoder()



y_train = encoder.fit_transform(train['sentiment'].values)

y_train = to_categorical(y_train) 



y_test = encoder.fit_transform(test['sentiment'].values)

y_test = to_categorical(y_test) 
# get an idea of the distribution of the text values

from collections import Counter

ctr = Counter(train['sentiment'].values)

print('Distribution of Classes:', ctr)



# get class weights for the training data, this will be used data

y_train_int = np.argmax(y_train,axis=1)

cws = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)

print(cws)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



np.set_printoptions(precision=4)

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """



    # Compute confusion matrix

    classes = classes[unique_labels(y_true, y_pred)]

    _cm = confusion_matrix(y_true, y_pred)



    print(classification_report(y_true, y_pred, target_names=classes))

        

    def _build_matrix(fig, ax, cm, normalize = False):

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'

        

        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



        im = ax.imshow(cm, cmap=cmap)

#         fig.colorbar(im, ax=ax)

        

        # We want to show all ticks...

        ax.set(xticks=np.arange(cm.shape[1]),

               yticks=np.arange(cm.shape[0]),

               # ... and label them with the respective list entries

               xticklabels=classes, 

               yticklabels=classes,

               title=title,

               ylabel='True label',

               xlabel='Predicted label')



        # Rotate the tick labels and set their alignment.

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



        # Loop over data dimensions and create text annotations.

        fmt = '.2f' if normalize else 'd'

        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(j, i, format(cm[i, j], fmt),

                        ha="center", va="center",

                        color="white" if cm[i, j] > thresh else "black")

        

    fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))

    _build_matrix(fig, ax1, cm = _cm, normalize=False)

    _build_matrix(fig, ax2, cm = _cm, normalize=True)

    fig.tight_layout()

# 

print('Dominant Class: ', ctr.most_common(n = 1)[0][0])

print('Baseline Accuracy Dominant Class', (ctr.most_common(n = 1)[0][0] == test['sentiment'].values).mean())



preds = np.zeros_like(y_test)

preds[:, 0] = 1

preds[0] = 1 #done to suppress warning from numpy for f1 score

print('F1 Score:', f1_score(y_test, preds, average='weighted'))

# Naive Bayse Baseline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline



text_clf = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', MultinomialNB()),

])

text_clf.fit(tokenizer.sequences_to_texts_generator(train_text_vec), y_train.argmax(axis=1))

predictions = text_clf.predict(tokenizer.sequences_to_texts_generator(test_text_vec)) 

print('Baseline Accuracy Using Naive Bayes: ', (predictions == y_test.argmax(axis = 1)).mean())

print('F1 Score:', f1_score(y_test.argmax(axis = 1), predictions, average='weighted'))



_ = plot_confusion_matrix(y_test.argmax(axis = 1), predictions, classes=encoder.classes_, title='Confusion matrix, without normalization')
# Random Forest Baseline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline



text_clf = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100)), #100 estimators will be the new default in version 0.22

])

text_clf.fit(tokenizer.sequences_to_texts_generator(train_text_vec), y_train.argmax(axis=1))

predictions = text_clf.predict(tokenizer.sequences_to_texts_generator(test_text_vec)) 

print('Baseline Accuracy Using RFC: ', (predictions == y_test.argmax(axis = 1)).mean())

print('F1 Score:', f1_score(y_test.argmax(axis = 1), predictions, average='weighted'))



_ = plot_confusion_matrix(y_test.argmax(axis = 1), predictions, classes=encoder.classes_)


def threshold_search(y_true, y_proba, average = None):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold, average=average)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result





def train(model, 

          X_train, y_train, X_test, y_test, 

          checkpoint_path='model.hdf5', 

          epcohs = 25, 

          batch_size = DEFAULT_BATCH_SIZE, 

          class_weights = None, 

          fit_verbose=2,

          print_summary = True

         ):

    m = model()

    if print_summary:

        print(m.summary())

    m.fit(

        X_train, 

        y_train, 

        #this is bad practice using test data for validation, in a real case would use a seperate validation set

        validation_data=(X_test, y_test),  

        epochs=epcohs, 

        batch_size=batch_size,

        class_weight=class_weights,

         #saves the most accurate model, usually you would save the one with the lowest loss

        callbacks= [

            ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True),

            EarlyStopping(patience = 2)

        ],

        verbose=fit_verbose

    ) 

    print("\n\n****************************\n\n")

    print('Loading Best Model...')

    m.load_weights(checkpoint_path)

    predictions = m.predict(X_test, verbose=1)

    print('Validation Loss:', log_loss(y_test, predictions))

    print('Test Accuracy', (predictions.argmax(axis = 1) == y_test.argmax(axis = 1)).mean())

    print('F1 Score:', f1_score(y_test.argmax(axis = 1), predictions.argmax(axis = 1), average='weighted'))

    plot_confusion_matrix(y_test.argmax(axis = 1), predictions.argmax(axis = 1), classes=encoder.classes_)

    plt.show()    

    return m #returns best performing model
def model_1():

    model = Sequential()

    model.add(Embedding(input_dim = (len(tokenizer.word_counts) + 1), output_dim = 128, input_length = MAX_SEQ_LEN))

    model.add(LSTM(128))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



m1 = train(model_1, 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           checkpoint_path='model_1.h5',

           class_weights=cws

          )

def model_1b():

    """

    Using a Bidiretional LSTM. 

    """

    model = Sequential()

    model.add(Embedding(input_dim = (len(tokenizer.word_counts) + 1), output_dim = 128, input_length = MAX_SEQ_LEN))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25)))

    model.add(Dense(64, activation='relu'))

#     model.add(Dropout(0.3))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



_ = train(model_1b, 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           checkpoint_path='model_1b.h5',

           class_weights=cws,

           print_summary = True

          )

def model_1c():

    """

    Adding dropout to reduce overfitting using a bidiretional LSTM

    """

    model = Sequential()

    model.add(Embedding(input_dim = (len(tokenizer.word_counts) + 1), output_dim = 128, input_length = MAX_SEQ_LEN))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Conv1D(64, 4))

#     model.add(Flatten())

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     print(model.summary())

    return model





_ = train(model_1c, 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           checkpoint_path='model_1c.h5',

           class_weights=cws,

           print_summary = True

          )

def model_1d():

    """

    Just for fun below is a model only using covolutions. This is pretty good and also trains very quickly(and predictions would also likely be fast) compared to the LSTM...

    It's equivalent to using an n-gram based approach.

    Usually in practice you would use a more complex architecture with multiple parallel convolutions that are combined before pooling(and usually both max and avg).

    Pure Convolutional NLP is definitely a solution worth exploring further.

    """

    model = Sequential()

    model.add(Embedding(input_dim = (len(tokenizer.word_counts) + 1), output_dim = 128, input_length = MAX_SEQ_LEN))

    model.add(SpatialDropout1D(0.3))

    model.add(Conv1D(64, 5))

    model.add(Conv1D(64, 3))

    model.add(Conv1D(64, 2))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





_ = train(model_1d, 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           checkpoint_path='model_1d.h5',

           class_weights=cws,

           print_summary = True

          )
def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')



def get_embdedings_matrix(embeddings_index, word_index, nb_words = None):

    all_embs = np.stack(embeddings_index.values())

    print('Shape of Full Embeddding Matrix', all_embs.shape)

    embed_dims = all_embs.shape[1]

    emb_mean,emb_std = all_embs.mean(), all_embs.std()



    #best to free up memory, given the size, which is usually ~3-4GB in memory

    del all_embs

    if nb_words is None:

        nb_words = len(word_index)

    else:

        nb_words = min(nb_words, len(word_index))

    

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_dims))

    found_vectors = 0

    words_not_found = []

    for word, i in tqdm(word_index.items()):

        if i >= nb_words: 

            continue

        embedding_vector = None

        if word in embeddings_index:

            embedding_vector = embeddings_index.get(word)

        elif word.lower() in embeddings_index:

            embedding_vector = embeddings_index.get(word.lower())

        # for twitter check if the key is a hashtag

        elif '#'+word.lower() in embeddings_index:

            embedding_vector = embeddings_index.get('#'+word.lower())

            

        if embedding_vector is not None: 

            found_vectors += 1

            embedding_matrix[i] = embedding_vector

        else:

            words_not_found.append((word, i))



    print("% of Vectors found in Corpus", found_vectors / nb_words)

    return embedding_matrix, words_not_found
def load_glove(word_index):

#     print('Loading Glove')

    embed_file_path = '../input/glove840b300dtxt/glove.840B.300d.txt'

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embed_file_path)))

    print("Built Embedding Index:", len(embeddings_index))

    return get_embdedings_matrix(embeddings_index, word_index)



def load_twitter(word_index):

#     print('Loading Twitter')

    embed_file_path = '../input/glove-twitter-27b-200d-txt/glove.twitter.27B.200d.txt'

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embed_file_path)))

    print("Built Embedding Index:", len(embeddings_index))

    return get_embdedings_matrix(embeddings_index, word_index)

print('Loading Glove Model...')

glove_embed_matrix, words_not_found =  load_glove(tokenizer.word_index)
print('Loading Twitter Model...')

twitter_embed_matrix, words_not_found =  load_twitter(tokenizer.word_index)
def model_2(embed_matrix):

    """

    Extends model_1 with a glove embedding

    """

    model = Sequential()

    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = MAX_SEQ_LEN,  weights=[embed_matrix], trainable=False))

    model.add(LSTM(128))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





m2 = train(lambda : model_2(glove_embed_matrix), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           checkpoint_path='model_2.h5',

           class_weights=cws,

           fit_verbose = 2,

           print_summary = False

          )
def model_3(embed_matrix):

    """

    Extends model 1c, will be trained with multiple embeddings

    """

    model = Sequential()

    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = MAX_SEQ_LEN,  weights=[embed_matrix], trainable=False))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Conv1D(64, 4))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
print("Model 3 w/ Glove Embedding")

_ = train(lambda : model_3(glove_embed_matrix), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=0,

           print_summary = False



          )



print("\n++++++++++++++++++++++++++++++++++++++++++\n")



print("Model 3 w/ Twitter Embedding")

_ = train(lambda : model_3(twitter_embed_matrix), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=0,

           print_summary = False



          )



print("\n++++++++++++++++++++++++++++++++++++++++++\n")



print("Model 3 w/ Stacked Embedding")

_ = train(lambda : model_3(np.hstack((twitter_embed_matrix, glove_embed_matrix))), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=0,

           print_summary = False



          )



def model_4(embed_matrix):



    model = Sequential()

    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = MAX_SEQ_LEN,  weights=[embed_matrix], trainable=False))

    model.add(SpatialDropout1D(0.25))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Conv1D(64, 4))

    model.add(Conv1D(32, 4))

    model.add(Conv1D(16, 4))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





_ = train(lambda : model_4(np.hstack((twitter_embed_matrix, glove_embed_matrix))), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=2,

           print_summary = True

          )



def model_5(embed_matrix):

    """

    Extends Model 3, but makes the embedding trainable

    """

    model = Sequential()

    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = MAX_SEQ_LEN,  weights=[embed_matrix], trainable=True))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Conv1D(64, 4))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





m5 = train(lambda : model_5(np.hstack((twitter_embed_matrix, glove_embed_matrix))), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=2,

           print_summary = True

          )
def model_6(embed_matrix):

    """

    Extends Model 5 and adds another Bidirectional LSTM layer

    """

    model = Sequential()

    model.add(Embedding(input_dim = embed_matrix.shape[0], output_dim = embed_matrix.shape[1], input_length = MAX_SEQ_LEN,  weights=[embed_matrix], trainable=True))

    model.add(SpatialDropout1D(0.3))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

    model.add(Conv1D(64, 4))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model





m6 = train(lambda : model_6(np.hstack((twitter_embed_matrix, glove_embed_matrix))), 

           train_text_vec,

           y_train,

           test_text_vec,

           y_test,

           class_weights=cws,

           fit_verbose=2,

           print_summary = False

          )



preds = m6.predict(test_text_vec)

print('Prediction, based on highest class:', (preds.argmax(axis = 1) == y_test.argmax(axis = 1)).mean())

#print('Prediction, based on class > 0.5:', ((y_test * preds).max(axis = 1) > 0.5).mean())



# Also consider searching the threshold, though this requires re-thinking the results, since you're now outputting up to 2 options for a class

# But should help with calling out ambiguous cases

threshold = threshold_search(y_test, preds, average='weighted')

print('Threshold Search:', threshold)

# print('Prediction, after Threshold Search:', (preds.argmax > threshold == y_test.argmax(axis = 1)).mean())
from sklearn.metrics import confusion_matrix

print('Residuals Analysis:', )

print(confusion_matrix(y_test.argmax(axis = 1),preds.argmax(axis = 1)))





ctr = 0

for i in range(y_test.shape[0]):

    true_label = y_test[i].argmax()

    pred_label = preds[i].argmax()

    if true_label != pred_label:

        print('idx:', i)

        print('True Label:', encoder.classes_[true_label])

        print('Predicted Label:', encoder.classes_[pred_label])

        print('Probability Prediction', preds[i])

        print(test['sentiment'].values[i], '::',  test['text'].values[i], '\n')

        ctr += 1

    

    if ctr > 20:

        break

        

words_not_found[:20]