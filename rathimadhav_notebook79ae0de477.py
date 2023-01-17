# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session//
#for pre-processing

import nltk

import re

from nltk.corpus import stopwords

from gensim.models import Word2Vec



nltk.download('stopwords')



#for model training



from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten

from keras.models import Sequential, load_model, model_from_config

import keras.backend as K





from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.metrics import cohen_kappa_score



test_data = pd.read_csv("/kaggle/input/asap-aes/test_set.tsv",sep='\t', encoding='ISO-8859-1')

training_data = pd.read_csv("/kaggle/input/asap-aes/training_set_rel3.tsv",sep='\t', encoding='ISO-8859-1',

                            usecols = ['essay_id', 'essay_set', 'essay','domain1_score']).dropna(axis=1)

valid_data = pd.read_csv("/kaggle/input/asap-aes/valid_set.tsv",sep='\t', encoding='ISO-8859-1')
test_data.dropna(axis=1,inplace=True)

valid_data.dropna(axis=1,inplace=True)
training_data

y = training_data['domain1_score']

X = training_data.copy()

X,y
def essay_to_wordlist(essay_v, remove_stopwords):

    #Remove the tagged labels and word tokenize the sentence.

    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)

    words = essay_v.lower().split()

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]

    return (words)



def essay_to_sentences(essay_v, remove_stopwords):

    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    raw_sentences = tokenizer.tokenize(essay_v.strip())

    sentences = []

    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0:

            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))

    return sentences



def makeFeatureVec(words, model, num_features):

    """Make Feature Vector from the words list of an Essay."""

    featureVec = np.zeros((num_features,),dtype="float32")

    num_words = 0.

    index2word_set = set(model.wv.index2word)

    for word in words:

        if word in index2word_set:

            num_words += 1

            featureVec = np.add(featureVec,model[word])        

    featureVec = np.divide(featureVec,num_words)

    return featureVec



def getAvgFeatureVecs(essays, model, num_features):

    """Main function to generate the word vectors for word2vec model."""

    counter = 0

    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")

    for essay in essays:

        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)

        counter = counter + 1

    return essayFeatureVecs
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten

from keras.models import Sequential, load_model, model_from_config

import keras.backend as K



def get_model():

    """Define the model."""

    model = Sequential()

    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))

    model.add(LSTM(64, recurrent_dropout=0.4))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='relu'))



    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])

    model.summary()



    return model
cv = KFold(n_splits = 5, shuffle = True)

results = []

y_pred_list = []



count = 1

for traincv, testcv in cv.split(X):

    print("\n--------Fold {}--------\n".format(count))

    X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    

    train_essays = X_train['essay']

    test_essays = X_test['essay']

    

    sentences = []

    

    for essay in train_essays:

            # Obtaining all sentences from the training essays.

            sentences += essay_to_sentences(essay, remove_stopwords = True)

            

    # Initializing variables for word2vec model.

    num_features = 300

    min_word_count = 40

    num_workers = 8

    context = 10

    downsampling = 1e-3



    print("Training Word2Vec Model...")

    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)



    model.init_sims(replace=True)

    model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)



    clean_train_essays = []

    

    # Generate training and testing data word vectors.

    for essay_v in train_essays:

        clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))

    trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)

    

    clean_test_essays = []

    for essay_v in test_essays:

        clean_test_essays.append(essay_to_wordlist( essay_v, remove_stopwords=True ))

    testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )

    

    trainDataVecs = np.array(trainDataVecs)

    testDataVecs = np.array(testDataVecs)

    # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)

    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))

    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    

    lstm_model = get_model()

    lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=2)

    #lstm_model.load_weights('./model_weights/final_lstm.h5')

    y_pred = lstm_model.predict(testDataVecs)

    

    # Save any one of the 5 models.

    if count == 5:

         lstm_model.save_weights('final_lstm.h5')

    

    # Round y_pred to the nearest integer.

    y_pred = np.around(y_pred)

    

    # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"

    result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')

    print("Kappa Score: {}".format(result))

    results.append(result)



    count += 1
print("Average Kappa score after a 5-fold cross validation: ",np.around(np.array(results).mean(),decimals=4))