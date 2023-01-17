import pandas as pd

import numpy as np

import re

np.random.seed(500) 

train=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/train.csv'))

train.head()

train=train.dropna(axis=1)

tweets=train['text'].to_list()

# print(train)

# print(tweets)

nonums=[]

for tweet in tweets:

    nonums.append(re.sub(r'\d+', '', tweet))

# print(nonums)
!pip install contractions

p=re.compile(r'\<http.+?\>', re.DOTALL)



tweetswithouturls=[]

for tweet in nonums:

    tweetswithouturls.append(re.sub(r"http\S+", "", tweet))

# print(tweetswithouturls)
import nltk

import contractions

def replace_contractions(text):

    """Replace contractions in string of text"""

    return contractions.fix(text)

nocontractions=[]

for tweet in tweetswithouturls:

    



    nocontractions.append(replace_contractions(tweet))

# print(nocontractions)
from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

tokens = [word_tokenize(sen) for sen in nocontractions]

# print(tokens)
def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words

nopunct=[]

for listt in tokens:

    nopunct.append(remove_punctuation(listt))

# print(nopunct)
import string, unicodedata

def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words

onlyascii=[]

for listt in nopunct:

    onlyascii.append(remove_non_ascii(listt))

# print(onlyascii)
def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words

lower=[]

for listt in onlyascii:

    lower.append(to_lowercase(listt))

# print(lower)


def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



    

nostopwords=[]

for listt in lower:

    nostopwords.append(remove_stopwords(listt))

# print(nostopwords)

# print(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

from nltk import pos_tag

from nltk.corpus import wordnet as wn

import collections

tag_map = collections.defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

Final_words = []

word_Lemmatized = WordNetLemmatizer()

for entry in nostopwords:

#     print(entry)

    words=[]

    # Initializing WordNetLemmatizer()

    

    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only alphabets

        

        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

#         print(word_Final)

        words.append(word_Final)

    Final_words.append(words)

# print(Final_words)

# for entry in nostopwords:

#     print(pos_tag(entry))
train['Text_Final'] = [' '.join(sen) for sen in Final_words]

train['tokens'] = Final_words

disaster = []

notdisaster = []

for l in train['target']:

    if l == 0:

        disaster.append(0)

        notdisaster.append(1)

    elif l == 1:

        disaster.append(1)

        notdisaster.append(0)

train['Disaster']= disaster

train['Not a Disaster']= notdisaster



train = train[['Text_Final', 'tokens', 'target', 'Disaster', 'Not a Disaster']]

train.head()
#repeating for test

test=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/test.csv'))

test.head()

test=test.dropna(axis=1)

tweetstest=test['text'].to_list()

# print(train)

# print(tweets)

nonumstest=[]

for tweet in tweetstest:

    nonumstest.append(re.sub(r'\d+', '', tweet))

# print(nonums)





tweetswithouturlstest=[]

for tweet in nonumstest:

    tweetswithouturlstest.append(re.sub(r"http\S+", "", tweet))

# print(tweetswithouturlstest)



nocontractionstest=[]

for tweet in tweetswithouturlstest:

    

    nocontractionstest.append(replace_contractions(tweet))

# print(nocontractionstest)

tokenstest = [word_tokenize(sen) for sen in nocontractionstest]

# print(tokenstest)



nopuncttest=[]

for listt in tokenstest:

    nopuncttest.append(remove_punctuation(listt))

# print(nopuncttest)



onlyasciitest=[]

for listt in nopuncttest:

    onlyasciitest.append(remove_non_ascii(listt))

# print(onlyasciitest)



lowertest=[]

for listt in onlyasciitest:

    lowertest.append(to_lowercase(listt))

# print(lowertest)



nostopwordstest=[]

for listt in lowertest:

    nostopwordstest.append(remove_stopwords(listt))

# print(nostopwordstest)

Finalwordstest=[]

for entry in nostopwordstest:

#     print(entry)

    words=[]

    # Initializing WordNetLemmatizer()

    

    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only alphabets

        

        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

#         print(word_Final)

        words.append(word_Final)

    Finalwordstest.append(words)

# print(Finalwordstest)
test['Text_Final'] = [' '.join(sen) for sen in Finalwordstest]

test['tokens'] = Finalwordstest

test.head()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

all_training_words = [word for tokens in train["tokens"] for word in tokens]



# count_vect = CountVectorizer()

# X_train_counts = count_vect.fit_transform(all_training_words)

# print(X_train_counts)

training_sentence_lengths = [len(tokens) for tokens in train["tokens"]]

TRAINING_VOCAB = sorted(list(set(all_training_words)))

print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))

print("Max sentence length is %s" % max(training_sentence_lengths))

### for svm split data



##tfidf

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(train["Text_Final"])

print(X_train_counts.shape)





tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

X_train_tf = tf_transformer.transform(X_train_counts)

print(X_train_tf.shape)

all_test_words = [word for tokens in test["tokens"] for word in tokens]

test_sentence_lengths = [len(tokens) for tokens in test["tokens"]]

TEST_VOCAB = sorted(list(set(all_test_words)))

print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))

print("Max sentence length is %s" % max(test_sentence_lengths))
MAX_SEQUENCE_LENGTH = 50



from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D,MaxPooling1D, Embedding

from keras.layers.recurrent import LSTM

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model



tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)

tokenizer.fit_on_texts(train["Text_Final"].tolist())

training_sequences = tokenizer.texts_to_sequences(train["Text_Final"].tolist())

train_word_index = tokenizer.word_index

print("Found %s unique tokens." % len(train_word_index))

train_cnn_data = pad_sequences(training_sequences, 

                               maxlen=MAX_SEQUENCE_LENGTH)
test_sequences = tokenizer.texts_to_sequences(test["Text_Final"].tolist())

test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# print(test_cnn_data.shape)

X_test_counts = count_vect.transform(test['Text_Final'])

X_test_tf = tf_transformer.transform(X_test_counts)

print(X_test_counts.shape)

print(X_test_tf.shape)
# # import gensim.downloader as api

# # path = api.load("word2vec-google-news-300", return_path=True)

path='../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin'

from gensim import models

word2vec = models.KeyedVectors.load_word2vec_format(path, binary=True)

EMBEDDING_DIM = 300

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))

for word,index in train_word_index.items():

    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

print(train_embedding_weights.shape)
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):

    

    embedding_layer = Embedding(num_words,

                            embedding_dim,

                            weights=[embeddings],

                            input_length=max_sequence_length,

                            trainable=False)

    

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')

    embedded_sequences = embedding_layer(sequence_input)



    convs = []

    filter_sizes = [2,3,4,5,6]



    for filter_size in filter_sizes:

        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)

        l_pool = GlobalMaxPooling1D()(l_conv)

        convs.append(l_pool)





    l_merge = concatenate(convs, axis=1)



    x = Dropout(0.1)(l_merge)  

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.2)(x)



    

    

    preds = Dense(2, activation='sigmoid')(x)



    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['acc'])

#     model.summary()

    return model
label_names = ['Disaster', 'Not a Disaster']

y_train = train[label_names].values





# print(y_train)

x_train = train_cnn_data
from keras.callbacks import EarlyStopping

num_epochs = 4 #3 is enough but just testing

batch_size = 24



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# for i in range(5):

#     print('Trial-',i)

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 

                len(list(label_names)))



hist = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size,callbacks=[es])

predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

print(predictions)
labels = [1, 0]

prediction_labels=[]

for p in predictions:

    prediction_labels.append(labels[np.argmax(p)])

# print(prediction_labels)

i=1

# for p in prediction_labels:

#     print(i,'-',p)

#     i+=1
test['target']=prediction_labels

# print(test[['tokens','target']])

submissions=pd.DataFrame(pd.read_csv('../input/nlp-getting-started/sample_submission.csv'))

# submissions['target']=prediction_labels

# print(submissions)

# submissions.to_csv('/kaggle/working/submission.csv',index=False)
# submissions=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

# comparewithnb=pd.DataFrame(pd.read_csv('../input/comparewithnb/filename11.csv'))

# cwnb=comparewithnb['0'].to_list()

# print(len(cwnb))

# count=0

# mismatch=[]

# for i in range(3263):

#     if(cwnb[i]==prediction_labels[i]):

#         count+=1

#     else:

#         mismatch.append(i)

# print(count)

        
testlabels=pd.DataFrame(pd.read_csv('../input/testlabels2/submission.csv'))

labels=testlabels['target'].to_list()

count=0

mismatch=[]

for i in range(3263):

    if(labels[i]==prediction_labels[i]):

        count+=1

    else:

        mismatch.append(i)

print(count)



from sklearn.linear_model import SGDClassifier

from sklearn import svm,metrics



train_model=svm.SVC().fit(X_train_tf, train["target"].values)

predictions=train_model.predict(X_test_tf)



# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

# hist2=SVM.fit(X_train_tf,train_y)

# predictions_SVM = SVM.predict(X_test_tf)

##SVM

count=0

mismatch=[]

for i in range(3263):

    if(labels[i]==predictions[i]):

        count+=1

    else:

        mismatch.append(i)

print(count)

submissions['target']=predictions



submissions.to_csv('/kaggle/working/submission.csv',index=False)


