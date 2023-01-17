import warnings

warnings.filterwarnings('ignore')



# Modules for data manipulation

import numpy as np

import pandas as pd

import re



# Modules for visualization

import matplotlib.pyplot as plt

import seaborn as sb



# Tools for preprocessing input data

from bs4 import BeautifulSoup

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



# Tools for creating ngrams and vectorizing input data

from gensim.models import Word2Vec, Phrases



# Tools for building a model

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, Bidirectional

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import pad_sequences



# Tools for assessing the quality of model prediction

from sklearn.metrics import accuracy_score, confusion_matrix



import os

for file in os.listdir("../input"):

    print(file)
SMALL_SIZE = 12

MEDIUM_SIZE = 14

BIG_SIZE = 16

LARGE_SIZE = 20



params = {

    'figure.figsize': (16, 8),

    'font.size': SMALL_SIZE,

    'xtick.labelsize': MEDIUM_SIZE,

    'ytick.labelsize': MEDIUM_SIZE,

    'legend.fontsize': BIG_SIZE,

    'figure.titlesize': LARGE_SIZE,

    'axes.titlesize': MEDIUM_SIZE,

    'axes.labelsize': BIG_SIZE

}

plt.rcParams.update(params)
usecols = ['sentiment','review']

train_data = pd.read_csv(

    filepath_or_buffer='../input/word2vec-nlp-tutorial/labeledTrainData.tsv',

    usecols=usecols, sep='\t')

additional_data = pd.read_csv(

    filepath_or_buffer='../input/imdb-review-dataset/imdb_master_filtered.csv',

    sep='\t')[usecols]

unlabeled_data = pd.read_csv(

    filepath_or_buffer="../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv", 

    error_bad_lines=False,

    sep='\t')

submission_data = pd.read_csv(

    filepath_or_buffer="../input/word2vec-nlp-tutorial/testData.tsv",

    sep='\t')
datasets = [train_data, additional_data, submission_data, unlabeled_data]

titles = ['Train data', 'Additional data', 'Unlabeled train data', 'Submission data']

for dataset, title in zip(datasets,titles):

    print(title)

    dataset.info()

    display(dataset.head())

all_reviews = np.array([], dtype=str)

for dataset in datasets:

    all_reviews = np.concatenate((all_reviews, dataset.review), axis=0)

print('Total number of reviews:', len(all_reviews))
train_data = pd.concat((train_data, additional_data[additional_data.sentiment != -1]),

                       axis=0, ignore_index=True)

train_data.info()
plt.hist(train_data[train_data.sentiment == 1].sentiment,

         bins=2, color='green', label='Positive')

plt.hist(train_data[train_data.sentiment == 0].sentiment,

         bins=2, color='blue', label='Negative')

plt.title('Classes distribution in the train data', fontsize=LARGE_SIZE)

plt.xticks([])

plt.xlim(-0.5, 2)

plt.legend()

plt.show()
def clean_review(raw_review: str) -> str:

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review, "lxml").get_text()

    # 2. Remove non-letters

    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)

    # 3. Convert to lower case

    lowercase_letters = letters_only.lower()

    return lowercase_letters





def lemmatize(tokens: list) -> list:

    # 1. Lemmatize

    tokens = list(map(lemmatizer.lemmatize, tokens))

    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))

    # 2. Remove stop words

    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))

    return meaningful_words





def preprocess(review: str, total: int, show_progress: bool = True) -> list:

    if show_progress:

        global counter

        counter += 1

        print('Processing... %6i/%6i'% (counter, total), end='\r')

    # 1. Clean text

    review = clean_review(review)

    # 2. Split into individual words

    tokens = word_tokenize(review)

    # 3. Lemmatize

    lemmas = lemmatize(tokens)

    # 4. Join the words back into one string separated by space,

    # and return the result.

    return lemmas
counter = 0

REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')

stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()
all_reviews = np.array(list(map(lambda x: preprocess(x, len(all_reviews)), all_reviews)))

counter = 0
X_train_data = all_reviews[:train_data.shape[0]]

Y_train_data = train_data.sentiment.values

X_submission = all_reviews[125000: 150000]
train_data['review_lenght'] = np.array(list(map(len, X_train_data)))

median = train_data['review_lenght'].median()

mean = train_data['review_lenght'].mean()

mode = train_data['review_lenght'].mode()[0]
fig, ax = plt.subplots()

sb.distplot(train_data['review_lenght'], bins=train_data['review_lenght'].max(),

            hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,

            kde_kws={"color": "black", 'linewidth': 3})

ax.set_xlim(left=0, right=np.percentile(train_data['review_lenght'], 95))

ax.set_xlabel('Words in review')

ymax = 0.014

plt.ylim(0, ymax)

ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)

ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)

ax.plot([median, median], [0, ymax], '--',

        label=f'median = {median:.2f}', linewidth=4)

ax.set_title('Words per review distribution', fontsize=20)

plt.legend()

plt.show()
%%time

bigrams = Phrases(sentences=all_reviews)
%%time

trigrams = Phrases(sentences=bigrams[all_reviews])
print(bigrams['space station near the solar system'.split()])
%%time

embedding_vector_size = 256

trigrams_model = Word2Vec(

    sentences = trigrams[bigrams[all_reviews]],

    size = embedding_vector_size,

    min_count=3, window=5, workers=4)
print("Vocabulary size:", len(trigrams_model.wv.vocab))
trigrams_model.wv.most_similar('galaxy')
trigrams_model.wv.doesnt_match(['galaxy', 'starship', 'planet', 'dog'])
%%time

def vectorize_data(data, vocab: dict) -> list:

    print('Vectorize sentences...', end='\r')

    keys = list(vocab.keys())

    filter_unknown = lambda word: vocab.get(word, None) is not None

    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))

    vectorized = list(map(encode, data))

    print('Vectorize sentences... (done)')

    return vectorized



print('Convert sentences to sentences with ngrams...', end='\r')

X_data = trigrams[bigrams[X_train_data]]

print('Convert sentences to sentences with ngrams... (done)')

input_length = 150

X_pad = pad_sequences(

    sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab),

    maxlen=input_length,

    padding='post')

print('Transform sentences to sequences... (done)')
X_train, X_test, y_train, y_test = train_test_split(

    X_pad,

    Y_train_data,

    test_size=0.05,

    shuffle=True,

    random_state=42)
def build_model(embedding_matrix: np.ndarray, input_length: int):

    model = Sequential()

    model.add(Embedding(

        input_dim = embedding_matrix.shape[0],

        output_dim = embedding_matrix.shape[1], 

        input_length = input_length,

        weights = [embedding_matrix],

        trainable=False))

    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))

    model.add(Dropout(0.25))

    model.add(Dense(64))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model



model = build_model(

    embedding_matrix=trigrams_model.wv.vectors,

    input_length=input_length)
model.compile(

    loss="binary_crossentropy",

    optimizer='adam',

    metrics=['accuracy'])



history = model.fit(

    x=X_train,

    y=y_train,

    validation_data=(X_test, y_test),

    batch_size=100,

    epochs=20)
def plot_confusion_matrix(y_true, y_pred, ax, class_names, vmax=None,

                          normed=True, title='Confusion matrix'):

    matrix = confusion_matrix(y_true,y_pred)

    if normed:

        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    sb.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,

               cmap=plt.cm.Blues_r, cbar=False, linecolor='black',

               linewidths=1, xticklabels=class_names)

    ax.set_title(title, y=1.20, fontsize=16)

    #ax.set_ylabel('True labels', fontsize=12)

    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)

    ax.set_yticklabels(class_names, rotation=0)
%%time

y_train_pred = model.predict_classes(X_train)

y_test_pred = model.predict_classes(X_test)
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2)

plot_confusion_matrix(y_train, y_train_pred, ax=axis1,

                      title='Confusion matrix (train data)',

                      class_names=['Positive', 'Negative'])

plot_confusion_matrix(y_test, y_test_pred, ax=axis2,

                      title='Confusion matrix (test data)',

                      class_names=['Positive', 'Negative'])
fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))



# summarize history for accuracy

axis1.plot(history.history['acc'], label='Train', linewidth=3)

axis1.plot(history.history['val_acc'], label='Validation', linewidth=3)

axis1.set_title('Model accuracy', fontsize=16)

axis1.set_ylabel('accuracy')

axis1.set_xlabel('epoch')

axis1.legend(loc='upper left')



# summarize history for loss

axis2.plot(history.history['loss'], label='Train', linewidth=3)

axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)

axis2.set_title('Model loss', fontsize=16)

axis2.set_ylabel('loss')

axis2.set_xlabel('epoch')

axis2.legend(loc='upper right')

plt.show()
print('Convert sentences to sentences with ngrams...', end='\r')

X_submit = trigrams[bigrams[X_submission]]

print('Convert sentences to sentences with ngrams... (done)')

X_sub = pad_sequences(

    sequences=vectorize_data(X_submit, vocab=trigrams_model.wv.vocab),

    maxlen=input_length,

    padding='post')

print('Transform sentences to sequences... (done)')
%%time

Y_sub_pred = model.predict_classes(X_sub)
def submit(predictions):

    submission_data['sentiment'] = predictions

    submission_data.to_csv('submission.csv', index=False, columns=['id','sentiment'])



submit(Y_sub_pred)