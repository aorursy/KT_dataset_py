!pip install PySastrawi

import numpy as np

import pandas as pd



import gc

gc.enable()

def run_gc():

    gc.collect() # avoid printing GC collect in output



import os

import multiprocessing

print(os.listdir('../input'))
# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/rupiah-movement-prediction-using-news/dataset.csv')

run_gc()

stopwords = pd.read_csv('../input/indonesian-stoplist/stopwordbahasa.csv', names=['stopword'])['stopword'].tolist()

run_gc()
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



#create stopword removal

factory = StopWordRemoverFactory()

stopword = StopWordRemover(ArrayDictionary(factory.get_stop_words() + stopwords))

run_gc()



#create stemmer

stem = StemmerFactory()

stemmer = stem.create_stemmer()

run_gc()
def pre_processing(text):

    clean_str = text.lower() # lowercase

    clean_str = re.sub(r"(?:\@|#|https?\://)\S+", " ", clean_str) # eliminate username, url, hashtags

    clean_str = re.sub(r'&amp;', '', clean_str) # remove &amp; as it equals &

    clean_str = re.sub(r'[^\w\s]',' ', clean_str) # remove punctuation

    clean_str = re.sub('[\s\n\t\r]+', ' ', clean_str) # remove extra space

    clean_str = clean_str.strip() # trim

    clean_str = " ".join([stemmer.stem(word) for word in clean_str.split()]) # stem

    clean_str = stopword.remove(clean_str) # remove stopwords

    return clean_str
data["clean"] = data["berita"].map(pre_processing, na_action='ignore')

run_gc()
def convert_to_label(delta):

    if not isinstance(delta, float) or delta == 0: return "stabil"

    if delta < 0: return "turun"

    return "naik"
data["label"] = data["delta"].map(convert_to_label)

run_gc()
data.head()
data.describe()
X = data["clean"]

y = data["label"]

run_gc()
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import word_tokenize
def tokenize_text(text):

    tokens = []

    for word in word_tokenize(text):

        if len(word) < 2:

            continue

        tokens.append(word.lower())

    return tokens
def doc2vec_transform(X, y, train_index, test_index):

    model = Doc2Vec(dm=1, dm_mean=1, vector_size=300,

                    window=10, negative=5, min_count=1,

                    workers=multiprocessing.cpu_count(),

                    alpha=0.065, min_alpha=0.065)

    y_train = y[train_index].tolist()

    train_tagged = [TaggedDocument(

        words=tokenize_text(_d),

        tags=[y_train[i]]

    ) for i, _d in enumerate(X[train_index])]

    model.build_vocab(train_tagged)

    run_gc()

    

    y_test = y[test_index].tolist()

    test_tagged = [TaggedDocument(

        words=tokenize_text(_d),

        tags=[y_test[i]]

    ) for i, _d in enumerate(X[test_index])]

    run_gc()

    

    return model, train_tagged, test_tagged
def vec_for_learning(model, tagged_docs):

    targets, classifiers = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in tagged_docs])

    return np.array(targets), np.array(classifiers)
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit

from sklearn.neural_network import MLPClassifier

from keras.wrappers.scikit_learn import KerasClassifier

import statistics
def train_classification(model, X_train, y_train, X_test, y_test):

    print(model)

    model.fit(X_train, y_train)

    

    result = model.predict(X_test)

    print("Result:", result)

    run_gc()

    

    CM = confusion_matrix(y_test, result)

    score = model.score(X_test, y_test)

    print("Test Result:")

    print("Confusion Matrix:")

    print(CM)

    print("Score:", score)

    test = score

    run_gc()

    

    self_res = model.predict(X_train)

    print("Self-Predict Result:", self_res)

    run_gc()

    

    CM = confusion_matrix(y_train, self_res)

    score = model.score(X_train, y_train)

    print("Self-Predict Test Result:")

    print("Confusion Matrix:")

    print(CM)

    print("Score:", score)

    self_test = score

    run_gc()

    

    return (test, self_test)
from keras.models import Sequential

from keras.layers import Dense

 

# Function to create model, required for KerasClassifier

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(1000, input_dim=300, kernel_initializer='normal', activation='relu')) # batch size 1000, vector size 300

    model.add(Dense(3, kernel_initializer='normal', activation='softmax')) # 3 classes: naik, turun, stabil

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
tscv = TimeSeriesSplit(n_splits=5)

classifiers = [{"model": x, "self_tests": [], "tests": []} for x in [

    KerasClassifier(build_fn=create_model, epochs=500, batch_size=1000, verbose=0),

    MLPClassifier(max_iter=1000, batch_size=1000, learning_rate='adaptive')

]]



for train_index, test_index in tscv.split(X):

    print("="*20)

    print("Train:", train_index, "Test:", test_index)

    d2v_model, train_tagged, test_tagged = doc2vec_transform(X, y, train_index, test_index)

    run_gc()

    y_train, X_train = vec_for_learning(d2v_model, train_tagged)

    y_test, X_test = vec_for_learning(d2v_model, test_tagged)

    run_gc()

    

    for classifier in classifiers:

        print("-"*20)

        test, self_test = train_classification(classifier["model"], X_train, y_train, X_test, y_test)

        classifier["self_tests"].append(self_test)

        classifier["tests"].append(test)

        run_gc()

    

print("="*20)

print("Final Result")

for classifier in classifiers:

    print("-"*20)

    print(classifier["model"])

    print("Self-Test Average Score:", statistics.mean(classifier["self_tests"]))

    print("Test Average Score:", statistics.mean(classifier["tests"]))