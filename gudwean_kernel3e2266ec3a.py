# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

 

os.environ["CUDA_VISIBLE_DEVICES"]="0";  

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from sklearn import preprocessing

from keras.preprocessing import sequence



from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding, SpatialDropout1D, MaxPooling1D, Embedding, Conv1D, Flatten, Dropout

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM

from keras.optimizers import Adam

from sklearn.utils import shuffle

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_extraction.text import CountVectorizer

import re



# load data

input_file = "../input/imdb-review-dataset/imdb_master.csv"



# comma delimited is the default

data = pd.read_csv(input_file, header = 0, encoding='ISO-8859-1', engine='python')
def rem_sw(df):

    # Downloading stop words

    stop_words = set(stopwords.words('english'))



    # Removing Stop words from training data

    count = 0

    for sentence in df:

        sentence = [word for word in sentence.lower().split() if word not in stop_words]

        sentence = ' '.join(sentence)

        df.loc[count] = sentence

        count+=1

    return(df)
def rem_punc(df):

    count = 0

    for s in df:

        cleanr = re.compile('<.*?>')

        s = re.sub(r'\d+', '', s)

        s = re.sub(cleanr, '', s)

        s = re.sub("'", '', s)

        s = re.sub(r'\W+', ' ', s)

        s = s.replace('_', '')

        df.loc[count] = s

        count+=1

    return(df)
def lemma(df):



    lmtzr = WordNetLemmatizer()



    count = 0

    stemmed = []

    for sentence in df:    

        word_tokens = word_tokenize(sentence)

        for word in word_tokens:

            stemmed.append(lmtzr.lemmatize(word))

        sentence = ' '.join(stemmed)

        df.iloc[count] = sentence

        count+=1

        stemmed = []

    return(df)
def stemma(df):



    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)



    count = 0

    stemmed = []

    for sentence in df:

        word_tokens = word_tokenize(sentence)

        for word in word_tokens:

            stemmed.append(stemmer.stem(word))

        sentence = ' '.join(stemmed)

        df.iloc[count] = sentence

        count+=1

        stemmed = []

    return(df)
df_master = pd.read_csv("../input/imdb-review-dataset/imdb_master.csv", encoding='latin-1', index_col = 0)



imdb_train = df_master[["review", "label"]][df_master.type.isin(['train'])].reset_index(drop=True)

imdb_test = df_master[["review", "label"]][df_master.type.isin(['test'])].reset_index(drop=True)



imdb_train['review'] = rem_sw(imdb_train['review'])



imdb_test['review'] = rem_sw(imdb_test['review'])



imdb_train['review'] = rem_punc(imdb_train['review'])



imdb_test['review'] = rem_punc(imdb_test['review'])



imdb_train['review'] = lemma(imdb_train['review'])

imdb_train['review'] = stemma(imdb_train['review'])



imdb_test['review'] = lemma(imdb_test['review'])

imdb_test['review'] = stemma(imdb_test['review'])
imdb_unsup = df_master[["review", "label"]][df_master.label.isin(['unsup'])].reset_index(drop=True)



# Cleaning Unlabelled data



imdb_unsup['review'] = rem_sw(imdb_unsup['review'])

imdb_unsup['review'] = rem_punc(imdb_unsup['review'])

imdb_unsup['review'] = lemma(imdb_unsup['review'])

imdb_unsup['review'] = stemma(imdb_unsup['review'])



# Vectorizing unlabelled reviews set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(imdb_unsup.review)



# Creating a dataframe for the high frequency words for unlabelled reviews set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T



# Removing high frequency and low frequency data for more accuracy

word_list = df_freq.nlargest(100, 'frequency').index

word_list = word_list.append(df_freq.nsmallest(43750, 'frequency').index)



# Removing unwanted words based on word_list from unlabelled data

count = 0

for sentence in imdb_unsup['review']:

    sentence = [word for word in sentence.lower().split() if word not in word_list]

    sentence = ' '.join(sentence)

    imdb_unsup.loc[count, 'review'] = sentence

    count+=1
################################## Preparing dataframe for model ##############################



# Creating df_algo dataframe which will be used for hypothesis testing

df_algo = pd.concat([imdb_train, imdb_test], keys=['train', 'test'])

df_algo = df_algo.reset_index(col_level=1).drop(['level_1'], axis=1)



# Cleaning the dataset

df_algo['review'] = rem_sw(df_algo['review'])

df_algo['review'] = rem_punc(df_algo['review'])

df_algo['review'] = lemma(df_algo['review'])

df_algo['review'] = stemma(df_algo['review'])



# df_algo = pd.read_csv("clean_algo.csv", encoding='latin-1', index_col = 0) # Uncomment this line to load from csv
################################### Removing non feature words ###############################

from sklearn.preprocessing import LabelEncoder

import operator



# Creating the feature word_list

# Selecting 14440 feature selected words based on 80-20 rule

le = LabelEncoder()

word_list = get_feature(df_algo[['review', 'label']], 14440)



# Removing non prefered words from training and test combined data

count = 0

for sentence in df_algo['review']:

    sentence = [word for word in sentence.lower().split() if word in word_list]

    sentence = ' '.join(sentence)

    df_algo.loc[count, 'review'] = sentence

    count+=1
################################## Splitting with feature selection data ###############################a

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorising the required data

vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')

vect_algo.fit(df_algo.review)

Xf_train = vect_algo.transform(df_algo[df_algo['level_0'].isin(['train'])].review)

Xf_test = vect_algo.transform(df_algo[df_algo['level_0'].isin(['test'])].review)



# Encoding target data

# Creating an object and fitting on target strings



yf_train = le.fit_transform(df_algo[df_algo['level_0'].isin(['train'])].label)

yf_test = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)
########################################### Naive Bayes #########################################

from sklearn.metrics import accuracy_score

# Fit the Naive Bayes classifier model to the object

clf = MultinomialNB()

clf.fit(Xf_train, yf_train)



# predict the outcome for testing data

predictions = clf.predict(Xf_test)



# check the accuracy of the model

accuracy = accuracy_score(yf_test, predictions)

print("Observation: Naive Bayes Classification gives an accuracy of %.2f%% on the testing data" %(accuracy*100))
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
data.head()
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 

sess = tf.Session(config=config) 

keras.backend.set_session(sess)
indexes_train = np.where(data['type'] == 'train')

indexes_test = np.where(data['type'] == 'test')



X_train = data['review'][indexes_train[0]].values

Y_train = data['label'][indexes_train[0]].values



X_test = data['review'][indexes_test[0]].values

Y_test = data['label'][indexes_test[0]].values



index_unsup = np.where(Y_train == 'unsup')

Y_train = np.delete(Y_train, index_unsup)

X_train = np.delete(X_train, index_unsup)
le = preprocessing.LabelEncoder()

le.fit(Y_train)



Y_train_encod = le.transform(Y_train) 

Y_test_encod = le.transform(Y_test) 



X_train, Y_train_encod = shuffle(X_train, Y_train_encod)
max_features = 10000

maxlen = 100

embedding_dimenssion = 100



VALIDATION_SPLIT = 0.1

CLASSES = 1

NB_EPOCH = 20

BATCH_SIZE = 64

OPTIMIZER = Adam(lr=0.001)



# Tokenization and encoding text corpus

tk = Tokenizer(num_words=max_features)

tk.fit_on_texts(X_train)

X_train_en = tk.texts_to_sequences(X_train)

X_test_en = tk.texts_to_sequences(X_test)



word2index = tk.word_index

index2word = tk.index_word
X_train_new = sequence.pad_sequences(X_train_en, maxlen=maxlen)

X_test_new = sequence.pad_sequences(X_test_en, maxlen=maxlen)



glove_dir = ''.join(['../input/glove6b/glove.6B.', str(embedding_dimenssion),'d.txt'])



embeddings_index = {}



with open(glove_dir, encoding='utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        embedding = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = embedding 

        

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))
embedding_matrix = np.zeros((max_features, embedding_dimenssion))



for word, i in word2index.items():

    if i >= max_features:

        break

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
model = Sequential()



model.add(Embedding(max_features, embedding_dimenssion, input_length=maxlen,

                    weights=[embedding_matrix], trainable=False))

model.add(LSTM(125, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation="sigmoid"))

model.summary()





model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])



model.fit(X_train_new, Y_train_encod, batch_size=BATCH_SIZE, epochs=10, validation_split=VALIDATION_SPLIT, verbose=1)



scores = model.evaluate(X_test_new, Y_test_encod)

print('losses: {}'.format(scores[0]))

print('TEST accuracy: {}'.format(scores[1]))