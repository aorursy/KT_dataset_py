import os

import re

import string
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk



from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer



from nltk.util import ngrams
# Vectorizers

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
train.shape
train.isna().sum()
train.target.unique()
test.shape
test.head()
text = train.text

target = train.target
def text_to_lowercase(text):

    """

    Convert text to lowercase

    """

    return str(text).lower()



def remove_number(text):

    """

    Remove numbers

    """

    result = re.sub(r'\d+', '', text)

    return result



def remove_punctuations(text):

    """

    Remove punctuation

    """

    for punctuation in string.punctuation:

        text = text.replace(punctuation, '')

    return text



def remove_whitespaces(text):

    """

    Remove whitespaces

    To remove leading and ending spaces

    """

    return text.strip()



def base_preparation(text):

    new_text = text_to_lowercase(text)

    new_text = remove_number(new_text)

    new_text = remove_punctuations(new_text)

    new_text = remove_whitespaces(new_text)

    return new_text
# base text preparation



text = text.apply(base_preparation)

test.text = test.text.apply(base_preparation)
# Tokenization



text = text.apply(word_tokenize)

test.text = test.text.apply(word_tokenize)
# Remove stop words



stop_words = set(stopwords.words('english'))



def remove_stop_words(text):

    return [i for i in text if not i in stop_words]



text = text.apply(remove_stop_words)

test.text = test.text.apply(remove_stop_words)
# Lemmatizer



wordnet_lemmatizer = WordNetLemmatizer()



def lemma(word_list):

    return [wordnet_lemmatizer.lemmatize(w) for w in word_list]



text = text.apply(lemma)

test.text = test.text.apply(lemma)
def do_nothing(tokens):

    return tokens



# TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(

    tokenizer=do_nothing, 

    preprocessor=None,

    lowercase=False,

    # ngram_range=(1, 2)

)
text_counts = tfidf_vectorizer.fit_transform(text)
X_train, X_test, y_train, y_test = train_test_split(

    text_counts, 

    target, 

    test_size=0.3, 

    random_state=1

)
print("X train shape: {0}".format(X_train.shape))

print("Y train shape: {0}".format(y_train.shape))

print("X test shape: {0}".format(X_test.shape))

print("Y test shape: {0}".format(y_test.shape))
from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV 

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import NearestCentroid

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier
def model_scoring(clf, X_test, y_test):

    predicted= clf.predict(X_test)

    print(classification_report(y_test, predicted))
%%time

# BernoulliNB



clf = BernoulliNB().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_bernouli_nb = round(clf.score(X_test, y_test) * 100, 2)
%%time

# DecisionTreeClassifier



clf = DecisionTreeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_decision_tree = round(clf.score(X_test, y_test) * 100, 2)
%%time

# ExtraTreesClassifier



clf = ExtraTreesClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_extra_tree = round(clf.score(X_test, y_test) * 100, 2)
%%time

# KNeighborsClassifier



clf = KNeighborsClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_knn = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LinearSVC  (setting multi_class=”crammer_singer”)



clf = LinearSVC(multi_class="crammer_singer").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_linear_svc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LogisticRegressionCV(setting multi_class=”multinomial”)



clf = LogisticRegressionCV(multi_class="multinomial").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_logistic_cv = round(clf.score(X_test, y_test) * 100, 2)
%%time

# MLPClassifier



clf = MLPClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_mlp = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RandomForestClassifier()



clf = RandomForestClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_random_forest = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RidgeClassifier



clf = RidgeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_ridge = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RidgeClassifierCV



clf = RidgeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_ridge_cv = round(clf.score(X_test, y_test) * 100, 2)
%%time

# SVC



clf = SVC().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_svc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# GradientBoostingClassifier



clf = GradientBoostingClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_gbc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LinearSVC



clf = LinearSVC(multi_class = "ovr").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_linear_svc2 = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LogisticRegression multi_class=”ovr”



clf = LogisticRegression(multi_class="ovr").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_logistic_reg = round(clf.score(X_test, y_test) * 100, 2)
%%time

# SGDClassifier



clf = SGDClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_sgd = round(clf.score(X_test, y_test) * 100, 2)
%%time

# Perceptron



clf = Perceptron().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_perceptron = round(clf.score(X_test, y_test) * 100, 2)
%%time

# PassiveAggressiveClassifier



clf = PassiveAggressiveClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_pac = round(clf.score(X_test, y_test) * 100, 2)
# evaluation



model_results = pd.DataFrame({

    'Models': [

        'BernoulliNB',

        'Decision Tree',

        'Extra Tree',

        'KNN',

        'Linear SVC',

        'Logistic Regression CV',

        'MLP',

        'Random Forest',

        'Ridge',

        'Ridge CV',

        'SVC',

        'GBC',

        'Linear SVC 2',

        'Logistic Regression',

        'SGDC',

        'Perceptron',

        'PAC'

    ],

    'Scores': [

        acc_bernouli_nb,

        acc_decision_tree,

        acc_extra_tree,

        acc_knn,

        acc_linear_svc,

        acc_logistic_cv,

        acc_mlp,

        acc_random_forest,

        acc_ridge,

        acc_ridge_cv,

        acc_svc,

        acc_gbc,

        acc_linear_svc2,

        acc_logistic_reg,

        acc_sgd,

        acc_perceptron,

        acc_pac

    ]

})

model_results.sort_values(by='Scores', ascending=False)
the_best_clf = BernoulliNB().fit(text_counts, target)
submition_example = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submition_example.head()
test_vectors = tfidf_vectorizer.transform(test.text)
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

submission(submission_file_path,the_best_clf,test_vectors)
import gensim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

cores = multiprocessing.cpu_count()
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from sklearn import utils
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
# base text preparation



train.text = train.text.apply(base_preparation)

test.text = test.text.apply(base_preparation)
# Tokenization



train.text = train.text.apply(word_tokenize)

test.text = test.text.apply(word_tokenize)
# Remove stop words



stop_words = set(stopwords.words('english'))



def remove_stop_words(text):

    return [i for i in text if not i in stop_words]



train.text = train.text.apply(remove_stop_words)

test.text = test.text.apply(remove_stop_words)
# Lemmatizer



wordnet_lemmatizer = WordNetLemmatizer()



def lemma(word_list):

    return [wordnet_lemmatizer.lemmatize(w) for w in word_list]



train.text = train.text.apply(lemma)

test.text = test.text.apply(lemma)
train_text, test_text = train_test_split(train, test_size=0.3, random_state = 42)
train_tagged = train_text.apply(

    lambda r: TaggedDocument(words=r['text'], tags=[r.target]), axis=1)

test_tagged = test_text.apply(

    lambda r: TaggedDocument(words=r['text'], tags=[r.target]), axis=1)
# Building a Vocabulary



model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)

model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
%%time

for epoch in range(10):

    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)

    model_dbow.alpha -= 0.002

    model_dbow.min_alpha = model_dbow.alpha
# Buliding the final vector feature for the classifier



def vec_for_learning(model, tagged_docs):

    sents = tagged_docs.values

    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])

    return targets, regressors
y_train, X_train = vec_for_learning(model_dbow, train_tagged)

y_test, X_test = vec_for_learning(model_dbow, test_tagged)
print("X train shape: {0}".format(np.array(X_train).shape))

print("Y train shape: {0}".format(np.array(y_train).shape))

print("X test shape: {0}".format(np.array(X_test).shape))

print("Y test shape: {0}".format(np.array(y_test).shape))
%%time

# BernoulliNB



clf = BernoulliNB().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_bernouli_nb = round(clf.score(X_test, y_test) * 100, 2)
%%time

# DecisionTreeClassifier



clf = DecisionTreeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_decision_tree = round(clf.score(X_test, y_test) * 100, 2)
%%time

# ExtraTreesClassifier



clf = ExtraTreesClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_extra_tree = round(clf.score(X_test, y_test) * 100, 2)
%%time

# KNeighborsClassifier



clf = KNeighborsClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_knn = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LinearSVC  (setting multi_class=”crammer_singer”)



clf = LinearSVC(multi_class="crammer_singer").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_linear_svc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LogisticRegressionCV(setting multi_class=”multinomial”)



clf = LogisticRegressionCV(multi_class="multinomial").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_logistic_cv = round(clf.score(X_test, y_test) * 100, 2)
%%time

# MLPClassifier



clf = MLPClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_mlp = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RandomForestClassifier()



clf = RandomForestClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_random_forest = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RidgeClassifier



clf = RidgeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_ridge = round(clf.score(X_test, y_test) * 100, 2)
%%time

# RidgeClassifierCV



clf = RidgeClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_ridge_cv = round(clf.score(X_test, y_test) * 100, 2)
%%time

# SVC



clf = SVC().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_svc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# GradientBoostingClassifier



clf = GradientBoostingClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_gbc = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LinearSVC



clf = LinearSVC(multi_class = "ovr").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_linear_svc2 = round(clf.score(X_test, y_test) * 100, 2)
%%time

# LogisticRegression multi_class=”ovr”



clf = LogisticRegression(multi_class="ovr").fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_logistic_reg = round(clf.score(X_test, y_test) * 100, 2)
%%time

# SGDClassifier



clf = SGDClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_sgd = round(clf.score(X_test, y_test) * 100, 2)
%%time

# Perceptron



clf = Perceptron().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_perceptron = round(clf.score(X_test, y_test) * 100, 2)
%%time

# PassiveAggressiveClassifier



clf = PassiveAggressiveClassifier().fit(X_train, y_train)

model_scoring(clf, X_test, y_test)



acc_pac = round(clf.score(X_test, y_test) * 100, 2)
model_results = pd.DataFrame({

    'Models': [

        'BernoulliNB',

        'Decision Tree',

        'Extra Tree',

        'KNN',

        'Linear SVC',

        'Logistic Regression CV',

        'MLP',

        'Random Forest',

        'Ridge',

        'Ridge CV',

        'SVC',

        'GBC',

        'Linear SVC 2',

        'Logistic Regression',

        'SGDC',

        'Perceptron',

        'PAC'

    ],

    'Scores': [

        acc_bernouli_nb,

        acc_decision_tree,

        acc_extra_tree,

        acc_knn,

        acc_linear_svc,

        acc_logistic_cv,

        acc_mlp,

        acc_random_forest,

        acc_ridge,

        acc_ridge_cv,

        acc_svc,

        acc_gbc,

        acc_linear_svc2,

        acc_logistic_reg,

        acc_sgd,

        acc_perceptron,

        acc_pac

    ]

})

model_results.sort_values(by='Scores', ascending=False)
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.preprocessing import text, sequence

from keras import utils

import tensorflow as tf
df = pd.read_csv('../input/nlp-getting-started/train.csv')
df.text = df.text.apply(base_preparation)
train_size = int(len(df) * .7)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(df) - train_size))
train_posts = df.text[:train_size]

train_tags = df.target[:train_size]



test_posts = df.text[train_size:]

test_tags = df.target[train_size:]
max_words = 10000

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)

x_test = tokenize.texts_to_matrix(test_posts)
encoder = LabelEncoder()

encoder.fit(train_tags)

y_train = encoder.transform(train_tags)

y_test = encoder.transform(test_tags)
num_classes = np.max(y_train) + 1

y_train = utils.to_categorical(y_train, num_classes)

y_test = utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
batch_size = 32

epochs = 10
# Build the model

model = Sequential()

model.add(Dense(512, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_split=0.1)
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print("Testing Accuracy:  {:.4f}".format(accuracy))
def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
plot_history(history)