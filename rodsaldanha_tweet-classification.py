# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from nltk.tokenize import word_tokenize

import en_core_web_sm

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics

from keras.models import Sequential

from keras import layers

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from keras.layers import Dropout

from keras import backend as K

from keras.callbacks import EarlyStopping

import re

import unicodedata

import nltk

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

from nltk.collocations import TrigramCollocationFinder

from nltk.metrics import TrigramAssocMeasures

from nltk.stem import WordNetLemmatizer 

import imblearn

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import ADASYN 

from imblearn.over_sampling import SMOTE
train_filepath = '../input/nlp-getting-started/train.csv'

test_filepath = '../input/nlp-getting-started/test.csv'



train = pd.read_csv(train_filepath, sep=",")

test = pd.read_csv(test_filepath, sep=",")

train.head()
train = train.fillna('')

test = test.fillna('')

#train['text'] = train[['keyword', 'location', 'text']].apply(lambda x: ' '.join(x), axis = 1)

train.head()
def normalize(txt):

    encoded_txt = []

    for word in txt.split():

        encoded_txt = [unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore') for word in txt.split()]

    new_txt = [word.decode() for word in encoded_txt] # decode

    new_txt_lower = [word.lower() for word in new_txt] # lower case

    return ' '.join(new_txt_lower)
# converts to plain text

train['text'] = train['text'].apply(lambda x: normalize(x)) 

test['text'] = test['text'].apply(lambda x: normalize(x))



train['keyword'] = train['keyword'].apply(lambda x: normalize(x)) 

test['keyword'] = test['keyword'].apply(lambda x: normalize(x))
def stop_w(txt):

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(txt) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    return ' '.join(filtered_sentence)
def replace(txt):

    replacers = {'dm': 'direct message',

                 'pm': 'private message',

                 'thx': 'thanks',

                 '2': '',

                 'b4': 'before',

                 'ab': 'about',

                 'abt': 'about',

                 'bfn': 'bye for now',

                 'btw': 'by the way',

                 'clk': 'click',

                 'chk': 'check',

                 'cld': 'could',

                 'da': 'the',

                 'eml': 'email',

                 'em': 'email',

                 'f2f': 'face-to-face',

                 'cya': 'see you',

                 'ic': 'i see',

                 'idk': 'i do not know',

                 'smh': 'shaking my head',

                 'sp':'sponsored',

                 'woz':'was',

                 'wtv':'whatever',

                 'gt': 'got',

                 '&amp;': 'and',

                 'amp;': 'and',

                 'amp': 'and',

                 '&gt;': '',

                 'plz': 'please',

                 'fyi': 'for your information',

                 'u': 'you',

                 'asap': 'as soon as possible',

                 '...': '',

                 '. . .': '',

                 'ur': 'your',

                 'n': 'and',

                 'cuz': 'because',

                 'w': 'with',

                 'im': 'i am',

                 'oh': '',

                 'eh': '',

                 'ah': '',

                 'tho': 'though',

                 'bc': 'because',

                 'wtf': 'what the fuck',

                 'st': 'street',

                 'til': 'until',

                 'ig': 'instagram',

                 'fb': 'facebook',

                 'afaik': 'as far as I know',

                 'br': 'best regards',

                 'dye': 'did you know',

                 'jk': 'just kidding',

                 'nb': 'no big deal',

                 'nbd': 'no big deal',

                 'nfw': 'no fucking way',

                 'nts': 'note to self',

                 'omg': 'my god',

                 'ty': 'thank you',

                 'lmao': 'lol',

                 'btwn': 'between',

                 'li': 'linkedin',

                 'we': 'weekend',

                 'omfg': 'my fucking god',

                 'wo': 'without',

                 'ppl': 'people',

                 'xoxo': 'kiss',

                 'aw': '',

                 'xo': 'kiss',

                 'mr': 'mister',

                 'ms': 'miss',

                 'haha': 'lol', 

                 'hah': 'lol',

                 'ha': 'lol',

                 'hwy': 'highway',

                 'rs': 'lol',

                 'nah': 'no',

                 'yeah': 'yes',

                 'yea': 'yes',

                 'ya': 'yes',

                 'yw': 'you are welcome',

                 'r': 'are'}

    

    splitLine=txt.split()

    for i in txt.split():

        if i in replacers :

            splitLine[splitLine.index(i)]=replacers[i]

            

    return ' '.join(splitLine)
train['text'] = train['text'].apply(lambda x: re.sub(r"http\S+", "",x)) # removes links

test['text'] = test['text'].apply(lambda x: re.sub(r"http\S+", "",x)) 



train['keyword'] = train['keyword'].apply(lambda x: re.sub(r"http\S+", "",x)) # removes links

test['keyword'] = test['keyword'].apply(lambda x: re.sub(r"http\S+", "",x)) 



train['text'] = train['text'].apply(lambda x: re.sub(r'\B@\w+', '',x)) # removes users

test['text'] = test['text'].apply(lambda x: re.sub(r'\B@\w+', '',x))



train['keyword'] = train['keyword'].apply(lambda x: re.sub(r'\B@\w+', '',x)) # removes users

test['keyword'] = test['keyword'].apply(lambda x: re.sub(r'\B@\w+', '',x))



train['text'] = train['text'].apply(lambda x: re.sub(r'(\w)(\1{2,})', r'\1',x)) # removes character repetition

test['text'] = test['text'].apply(lambda x: re.sub(r'(\w)(\1{2,})', r'\1',x))



train['keyword'] = train['keyword'].apply(lambda x: re.sub(r'(\w)(\1{2,})', r'\1',x)) # removes character repetition

test['keyword'] = test['keyword'].apply(lambda x: re.sub(r'(\w)(\1{2,})', r'\1',x))



train['text'] = train['text'].apply(lambda x: re.sub(r'[\W*]+', ' ',x)) # removes punctuation

test['text'] = test['text'].apply(lambda x: re.sub(r'[\W*]+', ' ',x)) 



train['keyword'] = train['keyword'].apply(lambda x: re.sub(r'[\W*]+', ' ',x)) # removes punctuation

test['keyword'] = test['keyword'].apply(lambda x: re.sub(r'[\W*]+', ' ',x)) 



train['text'] = train['text'].apply(lambda x: re.sub(r'\d+', '',x)) # removes numbers

test['text'] = test['text'].apply(lambda x: re.sub(r'\d+', '',x))



train['keyword'] = train['keyword'].apply(lambda x: re.sub(r'\d+', '',x)) # removes numbers

test['keyword'] = test['keyword'].apply(lambda x: re.sub(r'\d+', '',x))



train['text'] = train['text'].apply(lambda x: replace(x)) # vocabulary expansion

test['text'] = test['text'].apply(lambda x: replace(x))



train['keyword'] = train['keyword'].apply(lambda x: replace(x)) # vocabulary expansion

test['keyword'] = test['keyword'].apply(lambda x: replace(x))



train['text'] = train['text'].apply(lambda x: stop_w(x)) # remove stop words

test['text'] = test['text'].apply(lambda x: stop_w(x))



train['keyword'] = train['keyword'].apply(lambda x: stop_w(x)) # remove stop words

test['keyword'] = test['keyword'].apply(lambda x: stop_w(x))



train['text'].head()
def stemmer(txt):

    porter = nltk.PorterStemmer()

    word_tokens = word_tokenize(txt) 

    new_words = [porter.stem(w) for w in word_tokens]

    return ' '.join(new_words)
def lemmer(txt):

    lemmatizer = WordNetLemmatizer() 

    word_tokens = word_tokenize(txt) 

    print(word_tokens)

    new_words = [lemmatizer.lemmatize(w) for w in word_tokens]

    print(new_words)

    return ' '.join(new_words)
def replace_others(txt):

    replacers = {'th': 'number',

                 'uo': '',

                 'u_': '',

                 'na': '',

                 'gon': 'going',

                 'wan': 'want',

                 'pm': 'hour',

                 'x': '',

                 'p': ''

                }

    

    splitLine=txt.split()

    for i in txt.split():

        if i in replacers :

            splitLine[splitLine.index(i)]=replacers[i]

            

    return ' '.join(splitLine)
train['text'] = train['text'].apply(lambda x: replace_others(x)) 

test['text'] = test['text'].apply(lambda x: replace_others(x))



train['keyword'] = train['keyword'].apply(lambda x: replace_others(x)) 

test['keyword'] = test['keyword'].apply(lambda x: replace_others(x))



#train['text'] = train['text'].apply(lambda x: lemmer(x)) 

#test['text'] = test['text'].apply(lambda x: lemmer(x))



train['text'] = train['text'].apply(lambda x: stemmer(x))

test['text'] = test['text'].apply(lambda x: stemmer(x))



train['keyword'] = train['keyword'].apply(lambda x: stemmer(x))

test['keyword'] = test['keyword'].apply(lambda x: stemmer(x))



train['text'].head()
train.drop_duplicates(subset='text',inplace=True)
train['text'] = train[['keyword', 'text']].apply(lambda x: ' '.join(x), axis = 1) 

test['text'] = test[['keyword', 'text']].apply(lambda x: ' '.join(x), axis = 1)
from wordcloud import WordCloud

def print_wordcloud(data, bg_color):

    words = ' '.join(data)

    wordcloud = WordCloud(

                  background_color=bg_color,

                  width=3000,

                  height=2000

                ).generate(words)

    plt.figure(1, figsize=(15, 15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    

print_wordcloud(train['text'], 'black')
from nltk.probability import FreqDist



train_reviews = []

for index, row in train.iterrows():

    words_pre_process = [rvw for rvw in word_tokenize(row.text)]

    train_reviews.append((words_pre_process , row['target']))

    

def get_all_words(reviews):

    all_words = []

    for (words, label) in reviews:

        all_words.extend(words)

    return all_words



def get_bag_of_words(all_words):

    return nltk.FreqDist(all_words)
all_words = get_all_words(train_reviews)

bag_of_words = get_bag_of_words(all_words)

word_features = bag_of_words.keys()

bag_of_words.most_common(300)
def makeOverSamplesADASYN(X,y):

    sm = ADASYN()

    X, y = sm.fit_sample(X, y)

    return(X,y)
def makeOverSamplesSMOTE(X,y):

    sm = SMOTE()

    X, y = sm.fit_sample(X, y)

    return X,y
def underSample(X,y):

    rus = RandomUnderSampler(random_state=0)

    X, y = rus.fit_sample(X, y)

    return X,y
X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], 

                                                    test_size=0.2, random_state=1)



#vectorizer = TfidfVectorizer()

vectorizer = CountVectorizer()

vectorizer.fit(X_train)



X_predic = vectorizer.transform(test['text'])

X_train = vectorizer.transform(X_train)

X_test  = vectorizer.transform(X_test)



X_train,y_train = makeOverSamplesADASYN(X_train,y_train)

counter = Counter(y_train)

print(counter)
X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'], 

                                                    test_size=0.2, random_state=1)
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def create_model(X_train):

    input_dim = X_train.shape[1]  # Number of features

    

    model = Sequential()

    model.add(layers.Dense(200, input_dim=input_dim, activation='relu'))

    model.add(Dropout(0.4))

    model.add(layers.Dense(50, input_dim=input_dim, activation='relu'))

    model.add(Dropout(0.5))

    #model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=[f1_m])

    return model
model = create_model(X_train)



es_callback = EarlyStopping(monitor='val_loss', patience=5)



history = model.fit(X_train, y_train,

                    epochs=50,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    shuffle=True,

                    callbacks=[es_callback],

                    batch_size=10)
loss, f1_score = model.evaluate(X_test, y_test, verbose=False)

print("Training F1-score: {:.4f}".format(f1_score))
plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['f1_m']

    loss = history.history['loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training f1')

    plt.title('Training f1')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.title('Training loss')

    plt.legend()

    

plot_history(history)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../input/word2vec-google/GoogleNews-vectors-negative300.bin', binary=True)
train['tokens'] = train['text'].apply(lambda x: word_tokenize(x))

test['tokens'] = test['text'].apply(lambda x: word_tokenize(x))

train['tokens'].head()
X_train, X_test, y_train, y_test = train_test_split(train['tokens'], train['target'], 

                                                    test_size=0.2, random_state=1)
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)

matrix = vectorizer.fit_transform([x for x in X_train])

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print ('vocab size :', len(tfidf))
cores = multiprocessing.cpu_count() # Count the number of cores in a computer, important for a parameter of the model

w2v_model = Word2Vec(min_count=2,

                     window=2,

                     size=985,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)
from time import time

t = time()

w2v_model.build_vocab(train['tokens'], progress_per=1000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))



#TRAIN()

w2v_model.train(train['tokens'], total_examples=w2v_model.corpus_count, epochs=500, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
def buildWordVector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += w2v_model[word].reshape((1, size)) * tfidf[word]

            count += 1.

        except KeyError: # handling the case where the token is not

                         # in the corpus. useful for testing.

            continue

    if count != 0:

        vec /= count

    return vec
from sklearn.preprocessing import scale

X_train = np.concatenate([buildWordVector(z, 985) for z in map(lambda x: x, X_train)])

X_train = scale(X_train)



X_test = np.concatenate([buildWordVector(z, 985) for z in map(lambda x: x, X_test)])

X_test = scale(X_test)



X_predic = np.concatenate([buildWordVector(z, 985) for z in map(lambda x: x, test['tokens'])])

X_predic = scale(X_predic)



print ('shape for training set : ',X_train.shape,

      '\nshape for test set : ', X_test.shape)
X_train,y_train = makeOverSamplesSMOTE(X_train,y_train)

counter = Counter(y_train)

print(counter)
predictions = model.predict(X_predic)

predictions = [ int(x) for x in predictions ]
df=pd.DataFrame(test['id'])

df['target'] =predictions

df.set_index('id', inplace=True)

df.to_csv('submission.csv', sep=',')

df