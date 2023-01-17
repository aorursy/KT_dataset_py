import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import re

tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
# text preprocessing
contractions = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short

CUSTOM_FILTERS = [#lambda x: x.lower(), #lowercase
                  strip_tags, # remove html tags
                  #strip_punctuation, # replace punctuation with space
                  strip_multiple_whitespaces,# remove repeating whitespaces
                  strip_non_alphanum, # remove non-alphanumeric characters
                  #strip_numeric, # remove numbers
                  #remove_stopwords,# remove stopwords
                  strip_short # remove words less than minsize=3 characters long
                 ]
def gensim_preprocess(docs):
    docs = [expandContractions(doc) for doc in docs]
    docs = [preprocess_string(text, CUSTOM_FILTERS) for text in docs]
    docs = [' '.join(text) for text in docs]
    return pd.Series(docs)

train_clean = gensim_preprocess(train.question_text)

gensim_preprocess(train.question_text.iloc[10:15])
# creating vocab from train dataframe
from collections import Counter
vocab = Counter()

texts = ' '.join(train_clean).split()
vocab.update(texts)

print(len(vocab))
print(vocab.most_common(50))
# load google news vectors
from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)
# function to check coverage of embedding vs train vocabulary
import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
# function to correct misspellings and out of vocab words
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'Snapchat': 'social medium',
                'quora': 'social medium',
                'Quora': 'social medium',
                'mediumns': 'mediums',
                'bitcoin': 'currency',
                'cryptocurrency': 'currency',
                'upsc': 'union public service commission',
                'mbbs': 'bachelor medicine',
                'ece': 'educational credential evaluators',
                'aiims': 'all india institute medical science',
                'iim': 'india institute management',
                'sbi': 'state bank india',
                'blockchain': 'crytography',
                'and': '',
                'reducational':'educational',
                'neducational':'educational',
                'greeducational': 'greed educational',
                'pieducational': 'educational',
                'deducational': 'educational',
                'Quorans': 'Quoran'   
                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

# replace numbers > 9 with #### to match embedding
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

train_clean = train_clean.apply(lambda x: replace_typical_misspell(x))
train_clean = train_clean.apply(lambda x: clean_numbers(x))

vocab = Counter()
texts = ' '.join(train_clean).split()
vocab.update(texts)
# check out of vocab words again
oov = check_coverage(vocab,embeddings_index)
# view top 20 oov words
oov[:20]
# clean up our vocab
# keep tokens with a min occurrence
min_occurrence = 5
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))

vocab = set((' '.join(tokens)).split())
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# fit a tokenizer using keras
def create_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer
tokenizer = create_tokenizer(train_clean)
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen = max_length, padding='post')
    return padded
vocab_size = len(tokenizer.word_index) + 1
print('Vocab Size: ', vocab_size)
max_length = max([len(s.split()) for s in train_clean])
print('Max Length: ', max_length)
X_train = encode_docs(tokenizer, max_length, train_clean)
from keras.initializers import Constant
type(embeddings_index.vocab)
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 30000
word_index = tokenizer.word_index
num_words = vocab_size
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    try:
        embedding_vector = embeddings_index.get_vector(word)
    
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    except (KeyError):
        continue
        
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
# define and fit model on dataset
y_train = train.target

def define_model(vocab_size, max_length):
    #define model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = define_model(vocab_size, max_length)

def fit_model(X_train, y_train, epochs = 1):
    model.fit(X_train,
              y_train,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_split=0.1,
              class_weight={1:0.6, 0:0.4})
    return model
# create submodels
n_members = 5
members = []
for i in range(n_members):
    # fit model
    m = fit_model(X_train, y_train, epochs = (i+1))
    members.append(m)
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, X_train):
    stackX = None
    for model in members:
        # generate class probability prediction
        yhat = model.predict_proba(X_train, verbose=0)
        # stack predictions into [rows, predictions]
        if stackX == None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
            print(stackX.shape)
        # flatten predictions to [rows, predictions]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]))
        return stackX
# separate stacking model
from sklearn.linear_model import LogisticRegression

# fit model based on outputs from ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat
model = fit_stacked_model(members, X_train, y_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                y_train, test_size=0.2)

# evaluate model on test set
y_ = stacked_prediction(members, model, X_test)

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print(f1_score(y_test, y_))
print(classification_report(y_test, y_))
# prepare test data
test_clean = gensim_preprocess(test.question_text)
pred = encode_docs(tokenizer, max_length, test_clean)
# predict on test data
prediction = stacked_prediction(members, model, pred)
submission = pd.DataFrame({'qid':test.qid, 'prediction':prediction})
submission.to_csv('submission.csv', index=False)
submission.head()