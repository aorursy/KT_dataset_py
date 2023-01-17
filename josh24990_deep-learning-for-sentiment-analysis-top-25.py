# First up, I'll import every library that will be used in this project is imported at the start.

# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
import statsmodels.api as sm
from scipy.stats import randint as sp_randint
from time import time

# NLP
import nltk
nltk.download('wordnet')
import re
from textblob import TextBlob
from sklearn.feature_extraction import text
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Machine Learning
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# Data downloaded from Kaggle as a .csv file and read into this notebook from my local directory
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep = '\t')
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep = '\t')
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=',')
# General information about the Dataset
train.info()
# First 10 rows of the Dataset
train.head(10)
# Checking out the total number of unique sentences
train['SentenceId'].nunique()
# Returning average count of phrases per sentence, per Dataset
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
# Returning total phrase and sentence count, per Dataset
print('Number of phrases in train: {}; number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))
print('Number of phrases in test: {}; number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
# Returning average word length of phrases, per Dataset
print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
# Set up graph
fig, ax = plt.subplots(1, 1, dpi = 100, figsize = (10, 5))

# Get data
sentiment_labels = train['Sentiment'].value_counts().index
sentiment_count = train['Sentiment'].value_counts()

# Plot graph
sns.barplot(x = sentiment_labels, y = sentiment_count)

# Plot labels
ax.set_ylabel('Count')    
ax.set_xlabel('Sentiment Label')
ax.set_xticklabels(sentiment_labels , rotation=30)
# New column in the test set for concatenating
test['Sentiment']=-999
test.head()
# Concatenating Datasets before the cleaning can begin
data = pd.concat([train,test], ignore_index = True)
print(data.shape)
data.tail()
# Deleting previous Datasets from memory
del train,test
# Basic text cleaning function
def remove_noise(text):
    
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    
    # Convert to string
    text = text.astype(str)
        
    return text
# Apply the function and create a new column for the cleaned text
data['Clean Review'] = remove_noise(data['Phrase'])
data.head()
# Re-instating the training set
train = data[data['Sentiment'] != -999]
train.shape
# Re-instating the test set
test = data[data['Sentiment'] == -999]
test.drop('Sentiment', axis=1, inplace=True)
test.shape
# Getting a count of words from the documents
# Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
tokenizer = TweetTokenizer()

cvec = CountVectorizer(ngram_range=(1,2), tokenizer=tokenizer.tokenize)
full_text = list(train['Clean Review'].values) + list(test['Clean Review'].values)
cvec.fit(full_text)

# Getting the total n-gram count
len(cvec.vocabulary_)
# Creating the bag-of-words representation: training set
train_vectorized = cvec.transform(train['Clean Review'])

# Getting the matrix shape
print('sparse matrix shape:', train_vectorized.shape)

# Getting the nonzero count
print('nonzero count:', train_vectorized.nnz)

# Getting sparsity %
print('sparsity: %.2f%%' % (100.0 * train_vectorized.nnz / (train_vectorized.shape[0] * train_vectorized.shape[1])))
# Creating the bag-of-words representation: test set
test_vectorized = cvec.transform(test['Clean Review'])

# Getting the matrix shape
print('sparse matrix shape:', test_vectorized.shape)

# Getting the nonzero count
print('nonzero count:', test_vectorized.nnz)

# Getting sparsity %
print('sparsity: %.2f%%' % (100.0 * test_vectorized.nnz / (test_vectorized.shape[0] * test_vectorized.shape[1])))
# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
train_tdidf = transformer.fit_transform(train_vectorized)
test_tdidf = transformer.fit_transform(test_vectorized)
# Create X & y variables for Machine Learning
X_train = train_tdidf
y_train = train['Sentiment']

X_test = test_tdidf
# Model Fit and Prediction
def model(mod, model_name, X_train, y_train):
    
    # Fitting model
    mod.fit(X_train, y_train)
    
    # Print model name
    print(model_name)
    
    # Compute 5-fold cross validation: Accuracy
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 5)

    # Compute 5-fold prediction on training set
    predictions = cross_val_predict(mod, X_train, y_train, cv = 5)

    # Return accuracy score to 3dp
    print("Accuracy:", round(acc.mean(), 3))
 
    # Compute confusion matrix
    cm = confusion_matrix(predictions, y_train)
    
    # Print confusion matrix
    print("Confusion Matrix:  \n", cm)

    # Print classification report
    print("Classification Report \n", classification_report(predictions, y_train))
# Logistic Regression
log = LogisticRegression(multi_class='ovr')
model(log, "Logistic Regression", X_train, y_train)
# Importing all required tools for deep learning from keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
# 1. Tokenization
tokenizer = Tokenizer(lower=True, filters='')

tokenizer.fit_on_texts(full_text)
# 2. Indexing
train_sequences = tokenizer.texts_to_sequences(train['Clean Review'])
test_sequences = tokenizer.texts_to_sequences(test['Clean Review'])
# 3. Index Representation
MAX_LENGTH = 50

padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)
padded_train_sequences
# Find and plot total word count per sentence
totalNumWords = [len(one_comment) for one_comment in train_sequences]

plt.hist(totalNumWords,bins = np.arange(0,20,1))
plt.show()
# Setting max_len to 20 and padding data
max_len = 20
X_train = pad_sequences(train_sequences, maxlen = max_len)
X_test = pad_sequences(test_sequences, maxlen = max_len)
# Link to 2 million word vectors trained on Common Crawl (600B tokens)
embedding_path = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# Setting embedding size & max number of features
embed_size = 300
max_features = 30000
# Prepare the embedding matrix
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
        
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
# One hot encoding the y variable ready for deep learning application
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y_train.values.reshape(-1, 1))
# Create check-point
file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")

# Define callbacks
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

# Build model
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100,activation='relu') (x))
    x = Dense(5, activation = "sigmoid")(x)
    
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 15, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    
    return model
# Instantiating model
model = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)
# Making predictions with test data
pred = model.predict(X_test, batch_size = 1024)
# Preparing final submission file
predictions = np.round(np.argmax(pred, axis=1)).astype(int)

sub['Sentiment'] = predictions
sub.to_csv("submission.csv", index=False)