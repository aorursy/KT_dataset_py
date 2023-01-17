import warnings

warnings.filterwarnings('ignore')



# Modules for data manipulation

import numpy as np

import pandas as pd

import re



# Modules for visualization

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline



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

from keras.layers import Input, Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense, Bidirectional

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import pad_sequences



# Tools for assessing the quality of model prediction

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve



train_data = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = '\t')

unlabeled_data =  pd.read_csv("../input/unlabeledTrainData.tsv", error_bad_lines=False, delimiter='\t')

test_data = pd.read_csv("../input/testData.tsv", header = 0, delimiter = '\t')



# test_data["sentiment"] = test_data["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

# y_test = test_data["sentiment"]



datasets = [train_data, unlabeled_data, test_data]

titles = ['Train Data', 'Unlabeled Train Data', 'Test Data']

for dataset, title in zip(datasets,titles):

    print(title)

    dataset.info()

    display(dataset.head())
# Check class balance

plt.hist(train_data[train_data.sentiment == 1].sentiment,

         bins=2, color='green', label='Positive')

plt.hist(train_data[train_data.sentiment == 0].sentiment,

         bins=2, color='blue', label='Negative')

plt.title('Classes distribution in the train data')

plt.xticks([])

plt.xlim(-0.5, 2)

plt.legend()

plt.show()
# Define some pre-processing functions

def html_to_text(review):

    """Return extracted text string from provided HTML string."""

    review_text = BeautifulSoup(review, "lxml").get_text()

    if len(review_text) == 0:

        review_text = review

    review_text = re.sub(r"\<.*\>", "", review_text)

    try:

        review_text = review_text.encode('ascii', 'ignore').decode('ascii')#ignore \xc3 etc.

    except UnicodeDecodeError:

        review_text = review_text.decode("ascii", "ignore")

    return review_text





def letters_only(text):

    """Return input string with only letters (no punctuation, no numbers)."""

    # It is probably worth experimenting with milder prepreocessing (eg just removing punctuation)

    return re.sub("[^a-zA-Z]", " ", text)



def clean_review(review):

    """Preprocessing used before fitting/transforming RNN tokenizer - Html->text, remove punctuation/#s, lowercase."""

    return letters_only(html_to_text(review)).lower()



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
all_reviews = np.array([], dtype=str)

for dataset in datasets:

    all_reviews = np.concatenate((all_reviews, dataset.review), axis=0)

print('Total number of reviews:', len(all_reviews))



all_reviews = np.array(list(map(lambda x: preprocess(x, len(all_reviews)), all_reviews)))

counter = 0



X_train_data = all_reviews[:train_data.shape[0]]

Y_train_data = train_data.sentiment.values
train_data['review_length'] = np.array(list(map(len, X_train_data)))

median = train_data['review_length'].median()

mean = train_data['review_length'].mean()

mode = train_data['review_length'].mode()[0]



fig, ax = plt.subplots()

sb.distplot(train_data['review_length'], bins=train_data['review_length'].max(),

            hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,

            kde_kws={"color": "black", 'linewidth': 3})

ax.set_xlim(left=0, right=np.percentile(train_data['review_length'], 95))

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
bigrams = Phrases(sentences=all_reviews)

trigrams = Phrases(sentences=bigrams[all_reviews])

print(bigrams['space station near the solar system'.split()])



embedding_vector_size = 256

trigrams_model = Word2Vec(

    sentences = trigrams[bigrams[all_reviews]],

    size = embedding_vector_size,

    min_count=3, window=5, workers=4)

print("Vocabulary size:", len(trigrams_model.wv.vocab))
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

MAX_REVIEW_LENGTH = 150

X_pad = pad_sequences(sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab), maxlen=MAX_REVIEW_LENGTH, padding='post')

print('Transform sentences to sequences... (done)')
def build_model(embedding_matrix: np.ndarray, input_length: int,  use_lstm: bool):

    model = Sequential()

    model.add(Embedding(

        input_dim = embedding_matrix.shape[0],

        output_dim = embedding_matrix.shape[1], 

        input_length = input_length,

        weights = [embedding_matrix],

        trainable=False))

    if use_lstm:

        model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))

    else:

        model.add(Bidirectional(GRU(128, recurrent_dropout=0.1)))

    model.add(Dropout(0.25))

    model.add(Dense(64))

    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model



lstm_model = build_model(embedding_matrix=trigrams_model.wv.vectors, input_length=MAX_REVIEW_LENGTH, use_lstm=True)

lstm_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])



X_train, X_test, y_train, y_test = train_test_split(X_pad, Y_train_data, test_size=0.05, shuffle=True, random_state=42)

history = lstm_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=100, epochs=20)



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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from keras.layers import Input, Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense



gru_model = build_model(embedding_matrix=trigrams_model.wv.vectors, input_length=MAX_REVIEW_LENGTH, use_lstm=False)

gru_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

gru_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=100, epochs=20)



y_test_pred_lstm = lstm_model.predict(X_test)

y_test_pred_gru = gru_model.predict(X_test)



print("The AUC socre for GRU model is : %.4f." %roc_auc_score(y_test, y_test_pred_gru))

print("The AUC socre for LSTM model is : %.4f." %roc_auc_score(y_test, y_test_pred_lstm))



y_pred_list = [y_test_pred_gru, y_test_pred_lstm]

label_list = ["GRU", "LSTM"]

pred_label = zip(y_pred_list, label_list)

for y_pred, lbl in pred_label:

    fpr, tpr, _ = roc_curve(y_test, y_pred)

    plt.plot(fpr, tpr, label = lbl)



plt.xlabel("True Postive Rate")

plt.ylabel("False Positive Rate")

plt.title("ROC Curve for RNN Models")

plt.legend()

plt.show()
y_train_pred = gru_model.predict_classes(X_train)

y_test_pred = gru_model.predict_classes(X_test)



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

    

fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2)

plot_confusion_matrix(y_train, y_train_pred, ax=axis1,

                      title='Confusion matrix (train data)',

                      class_names=['Positive', 'Negative'])

plot_confusion_matrix(y_test, y_test_pred, ax=axis2,

                      title='Confusion matrix (test data)',

                      class_names=['Positive', 'Negative'])
print('Convert sentences to sentences with ngrams...', end='\r')

X_submission_data = all_reviews[-25000:]

X_submission = trigrams[bigrams[X_submission_data]]

print('Convert sentences to sentences with ngrams... (done)')

X_submission_pad = pad_sequences(

    sequences=vectorize_data(X_submission, vocab=trigrams_model.wv.vocab),

    maxlen=MAX_REVIEW_LENGTH,

    padding='post')

print('Transform sentences to sequences... (done)')



predictions = gru_model.predict_classes(X_submission_pad)



submission = pd.DataFrame()

submission['id'] = test_data['id']

submission['sentiment'] = predictions

submission.to_csv('submission.csv',index=False)