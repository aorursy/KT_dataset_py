# Library

import pandas as pd

import numpy as np

import spacy

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import chi2



import tensorflow as tf

from tensorflow import keras

import keras.backend as K



import warnings

warnings.filterwarnings('ignore')
# Read file

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# See number of samples per label

samples = train[['text', 'target']].groupby("target").count().reset_index()

samples['total'] = samples.text.sum()

samples['proportion'] = samples.text / samples.total

samples



# This dataset is not an imbalance dataset
# Count words / token in sample text

def len_token(x):

    return len(x.split())



train['num_words'] = train.text.apply(lambda x: len_token(x))
# See number of words per samples

plt.figure(figsize=(8,8))

sns.countplot(x = train['num_words'])

plt.xticks(rotation=90)



train['num_words'].describe()

# The average of num_words in samples is 17.7 words

# and the 3rd quatile is 23
# Difference on distribution num_words on label 1 and 0

sns.catplot(x='target', y = 'num_words', kind = 'violin', data=train )

train[['num_words', 'target']].groupby('target').describe()



# Label 0 tend to have higher num_words showed in 3rd quartile and its higher std compare to label 1
# Get ratio of samples per number of word per samples

train['num_words'].count() / (train['num_words'].sum() / train['num_words'].count())



# The dataset have ratio (samples) per (number of word per samples) = 510 < 1500

# Based on this post https://developers.google.com/machine-learning/guides/text-classification/step-2-5

# The n-gram model would be better and perform well, we don't have to use sequence model yet
# Split train and validation dataset

X_train, X_val, y_train, y_val = train_test_split(train, train.target,

                                                    test_size=0.1, random_state=1)



X_train.shape, X_val.shape, y_train.shape, y_val.shape
# See number of samples per label on the training dataset

# The samples has the same proportion with the whole training data

samples = y_train.value_counts().reset_index()

samples['total'] = samples.target.sum()

samples['proportion'] = samples.target / samples.total

samples
# Fit CountVectorizer and see the vocabulary

vectorizer = CountVectorizer()

vectorizer.fit(X_train['text'])

# vectorizer.vocabulary_
# Create vector from Count Vecorizer

train_vector = vectorizer.transform(X_train['text']) 

val_vector = vectorizer.transform(X_val['text'])
# Build simple model using Logistic Regression

logreg = LogisticRegression()



logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score Baseline model on validation dataset: {}".format(score))



# 0.735 is our baseline score
# Train model on all training data

# Test using test dataset

logreg = LogisticRegression()



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



logreg.fit(train_vector, train.target)

y_pred = logreg.predict(test_vector)
# Submission using baseline model

submission_baseline = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_baseline['target'] = y_pred

submission_baseline.to_csv('baseline_logreg.csv', index=False)
# Previously we are using CountVectorizer which using count of each word to generate vector representation of each instance.

# Now we are using TF-IDF Vectorizer to generate vector representation of the instances.

# The difference is TF-IDF will consider word Frequency and also Inverse Document (that the word appear on)Frequency

# The implementation is using sklearn



# Create vectorizer and fit with training data

tfidfvectorizer = TfidfVectorizer(ngram_range= (1,4), max_features=4000)

tfidfvectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = tfidfvectorizer.transform(X_train['text'])

val_vector = tfidfvectorizer.transform(X_val['text'])



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using TF-IDF on validation dataset: {}".format(score))

# The LogReg with TF-IDF with adjusted hyperparameter perform better than baseline model. Its f1 score is 0.7466
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

tfidfvectorizer = TfidfVectorizer(ngram_range= (1,4), max_features=4000)

tfidfvectorizer.fit(train['text'])



train_vector = tfidfvectorizer.transform(train['text'])

test_vector = tfidfvectorizer.transform(test['text'])



logreg = LogisticRegression()



logreg.fit(train_vector, train.target)

y_pred = logreg.predict(test_vector)
# Submission using baseline model

submission_tfidf = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_tfidf['target'] = y_pred

submission_tfidf.to_csv('tfidf_logreg.csv', index=False)
# Create vectorizer and fit with training data

vectorizer = HashingVectorizer(binary= True)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = vectorizer.transform(X_train['text'])

val_vector = vectorizer.transform(X_val['text'])



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using HashingVectorizer on validation dataset: {}".format(score))

# The Binary Vectorizer perform better than Count Vectorizer, but not TF-IDF Vectorizer

# The F1 Score is 0.74
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

vectorizer = HashingVectorizer(binary= True)

vectorizer.fit(train['text'])



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



logreg = LogisticRegression()



logreg.fit(train_vector, train.target)

y_pred = logreg.predict(test_vector)
# Submission using binary vectorizer

submission_binary = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_binary['target'] = y_pred

submission_binary.to_csv('binary_vect_logreg.csv', index=False)
# Create vectorizer and fit with training data

tfidfvectorizer = CountVectorizer(ngram_range= (1,4), max_features=4000)

tfidfvectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = tfidfvectorizer.transform(X_train['text'])

val_vector = tfidfvectorizer.transform(X_val['text'])



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using CountVectorizer with adjusted hyperparameter on validation dataset: {}".format(score))

# These adjusted hyperparameter help CountVectorizer to perform better. its f1 score is 0.7529
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,4), max_features=4000)

vectorizer.fit(train['text'])



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



logreg = LogisticRegression()



logreg.fit(train_vector, train.target)

y_pred = logreg.predict(test_vector)
# Submission using count vectorizer with adjusted hyperparameter

submission_better_count = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_better_count['target'] = y_pred

submission_better_count.to_csv('better_count_logreg.csv', index=False)
# Create vectorizer and fit with training data

tfidfvectorizer = CountVectorizer(ngram_range= (1,4), max_features=4000)

tfidfvectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = tfidfvectorizer.transform(X_train['text'])

val_vector = tfidfvectorizer.transform(X_val['text'])



# Standadize features

scaler = Normalizer()

scaler.fit(train_vector)



train_vector = scaler.transform(train_vector)

val_vector = scaler.transform(val_vector)



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using CountVectorizer with adjusted hyperparameter on validation dataset: {}".format(score))

# These adjusted hyperparameter help CountVectorizer and Normalize did not perform better. its f1 score is 0.728
# Now we have two possible vectorizer, we will try using difference modelling which is MLP

# We will use shallow MLP first to see whether the model will perform better or not than Logistic Regression



# MLP with Count Vectorizer



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features=4000)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_count_vector = vectorizer.transform(X_train['text'])

val_count_vector = vectorizer.transform(X_val['text'])



# MLP Architecture

mlp_count = keras.Sequential([

    keras.layers.Dense(1028, activation= 'relu', kernel_initializer= 'he_normal', input_shape=train_count_vector.shape[1:]),

    keras.layers.Dense(256, activation= 'relu', kernel_initializer= 'he_normal'),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp_count.compile(optimizer= 'adam',

                  loss= 'binary_crossentropy',

                  metrics= ['accuracy']

                 )



# Train MLP

mlp_count.fit(train_count_vector.toarray(), y_train, epochs=10)
# Evaluate model in validation dataset

test_loss, test_acc = mlp_count.evaluate(val_count_vector.toarray(),y_val, verbose= 0)

print("MLP Accuracy on Validation: {}".format(test_acc))



# Evaluate model in validation dataset

y_pred = mlp_count.predict_classes(val_count_vector.toarray())

score = f1_score(y_val, y_pred)

print("F1 Score MLP and Count Vectorizer on validation dataset: {}".format(score))
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features=4000)

vectorizer.fit(train['text'])



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



# MLP Architecture

mlp_count = keras.Sequential([

    keras.layers.Dense(1028, activation= 'relu', kernel_initializer= 'he_normal', input_shape=train_count_vector.shape[1:]),

    keras.layers.Dense(256, activation= 'relu', kernel_initializer= 'he_normal'),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp_count.compile(optimizer= 'adam',

                  loss= 'binary_crossentropy',

                  metrics= ['accuracy']

                 )



# Train MLP

mlp_count.fit(train_vector.toarray(), train.target, epochs=20, verbose=0)



# Prediction on test dataset

y_pred = mlp_count.predict_classes(test_vector.toarray())



# Submission using MLP and count vectorizer

submission_mlp_count = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_mlp_count['target'] = y_pred

submission_mlp_count.to_csv('count_mlp.csv', index=False)
# MLP with TF-IDF Vectorizer



# Create vectorizer and fit with training data

tfidfvectorizer = TfidfVectorizer(ngram_range= (1,2), max_features=4000)

tfidfvectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_tfidf_vector = tfidfvectorizer.transform(X_train['text'])

val_tfidf_vector = tfidfvectorizer.transform(X_val['text'])



# MLP Architecture

mlp_tfidf = keras.Sequential([

    keras.layers.Dense(1028, activation= 'relu', kernel_initializer= 'he_normal', input_shape=train_tfidf_vector.shape[1:]),

    keras.layers.Dense(256, activation= 'relu', kernel_initializer= 'he_normal'),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp_tfidf.compile(optimizer= 'adam',

                  loss= 'binary_crossentropy',

                  metrics= ['accuracy']

                 )



# Train MLP

mlp_tfidf.fit(train_tfidf_vector.toarray(), y_train, epochs=10)
# Evaluate model in validation dataset

test_loss, test_acc = mlp_tfidf.evaluate(val_count_vector.toarray(),y_val, verbose= 0)

print("MLP Accuracy on Validation: {}".format(test_acc))



# Evaluate model in validation dataset

y_pred = mlp_tfidf.predict_classes(val_count_vector.toarray())

score = f1_score(y_val, y_pred)

print("F1 Score MLP and TF-IDF Vectorizer on validation dataset: {}".format(score))
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

tfidfvectorizer = TfidfVectorizer(ngram_range= (1,2), max_features=4000)

vectorizer.fit(train['text'])



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



# MLP Architecture

mlp_tfidf = keras.Sequential([

    keras.layers.Dense(1028, activation= 'relu', kernel_initializer= 'he_normal', input_shape=train_vector.shape[1:]),

    keras.layers.Dense(256, activation= 'relu', kernel_initializer= 'he_normal'),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp_tfidf.compile(optimizer= 'adam',

                  loss= 'binary_crossentropy',

                  metrics= ['accuracy']

                 )



# Train MLP

mlp_tfidf.fit(train_vector.toarray(), train.target, epochs=10, verbose=0)



# Prediction on test dataset

y_pred = mlp_tfidf.predict_classes(test_vector.toarray())



# Submission using MLP and count vectorizer

submission_mlp_tfidf = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_mlp_tfidf['target'] = y_pred

submission_mlp_tfidf.to_csv('tfidf_mlp.csv', index=False)
# First we want to see whether eliminating stopwords will increase the model performance

stop_words = list(ENGLISH_STOP_WORDS)
# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,4), max_features = 4000, stop_words = stop_words)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = vectorizer.transform(X_train['text'])

val_vector = vectorizer.transform(X_val['text'])



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using Count Vectorizer with Stop Words on validation dataset: {}".format(score))

# Giving stop words worsen the model performance on F1 score
# Get the models coefficients (and top 20 and bottom 20)

def plot_top_bottom_20_coef(model, feature_names):

    logReg_coeff = pd.DataFrame({'feature_name': feature_names, 'model_coefficient': model.coef_.transpose().flatten()})

    logReg_coeff = logReg_coeff.sort_values('model_coefficient',ascending=False)

    logReg_coeff_top = logReg_coeff.head(20)

    logReg_coeff_bottom = logReg_coeff.tail(20)



    plt.figure().set_size_inches(10, 6)

    fg3 = sns.barplot(x='feature_name', y='model_coefficient',data=logReg_coeff_top, palette="Blues_d")

    fg3.set_xticklabels(rotation=35, labels=logReg_coeff_top.feature_name)

    # Plot bottom 5 coefficients

    plt.figure().set_size_inches(10,6)

    fg4 = sns.barplot(x='feature_name', y='model_coefficient',data=logReg_coeff_bottom, palette="GnBu_d")

    fg4.set_xticklabels(rotation=35, labels=logReg_coeff_bottom.feature_name)

    plt.xlabel('Feature')

    plt.ylabel('Coefficient')

    plt.subplots_adjust(bottom=0.4)
# Plot logistic regression coefficient on using Count Vectorizer

# get feature names

feature_names = np.array(vectorizer.get_feature_names())

plot_top_bottom_20_coef(logreg, feature_names)
# See coefficient Logistic Regression on TF-IDF Vectorizer



# Create vectorizer and fit with training data

tfidfvectorizer = TfidfVectorizer(ngram_range= (1,4), max_features=4000)

tfidfvectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = tfidfvectorizer.transform(X_train['text'])

val_vector = tfidfvectorizer.transform(X_val['text'])



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using TF-IDF on validation dataset: {}".format(score))

# Plot Coefficient TF-IDF LogReg Model



# get feature names

feature_names = np.array(tfidfvectorizer.get_feature_names())

plot_top_bottom_20_coef(logreg, feature_names)



# 20 positive coefficient features from Count Vectorizer and TF-IDF Vectorizer quite the same

# The difference is in negative coefficient. Count Vectorizer tend to have higher negative coefficient than TF-IDF

# Could we see which features that have the same correlation and drop it?
# Chi Square test is one of the way to see linear correlation between features to label



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,4), max_features = 4000)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = vectorizer.transform(X_train['text'])

val_vector = vectorizer.transform(X_val['text'])



# Convert vector to array

features = train_vector.toarray()

labels = y_train



# Get unigram and bigram the most correlated features

N = 10

features_chi2 = chi2(features, labels == labels)

indices = np.argsort(features_chi2[0])

feature_names = np.array(vectorizer.get_feature_names())[indices]

unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features= 4000)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = vectorizer.transform(X_train['text'])

val_vector = vectorizer.transform(X_val['text'])



# Select K best

selector = SelectKBest(f_classif, k = 3000)

selector.fit(train_vector, y_train)



# Select vector

train_vector = selector.transform(train_vector)

val_vector = selector.transform(val_vector)



# Build simple model using LogisticRegression

logreg = LogisticRegression()

logreg.fit(train_vector, y_train)

y_pred = logreg.predict(val_vector)



score = f1_score(y_val, y_pred)

print("F1 Score LogReg using CountVectorizer and SelectKBest with f_classif on validation dataset: {}".format(score))

# Selecting 3000 with f_classif improve f1 score to 0.762

# I tried to use chi2 to select features, but it does not give difference with f_classif
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features=4000)

vectorizer.fit(train['text'])



train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



# SelectKBest using f_classif

selector = SelectKBest(f_classif, k = 3000)

selector.fit(train_vector, train.target)



# Select vector

train_vector = selector.transform(train_vector)

test_vector = selector.transform(test_vector)



logreg = LogisticRegression()



logreg.fit(train_vector, train.target)

y_pred = logreg.predict(test_vector)



# Submission using count vectorizer with adjusted hyperparameter

submission_count_selectk = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_count_selectk['target'] = y_pred

submission_count_selectk.to_csv('count_selectk_logreg.csv', index=False)
# Previously we are optimizing the model using accuracy metric instead of f1_score

# We now will implement custom metric f1_score for optimizing the model



def f1(y_true, y_pred):

    # Count positive samples.

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))



    # How many selected items are relevant?

    precision = c1 / c2



    # How many relevant items are selected?

    recall = c1 / c3



    # Calculate f1_score

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score 
# MLP with Select K Best



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features= 4000)

vectorizer.fit(X_train['text'])



# Create vector of train and validation data using tfidf

train_vector = vectorizer.transform(X_train['text'])

val_vector = vectorizer.transform(X_val['text'])



# Select K best

selector = SelectKBest(f_classif, k = 3000)

selector.fit(train_vector, y_train)



# Select vector

train_vector = selector.transform(train_vector)

val_vector = selector.transform(val_vector)



# MLP Architecture

mlp = keras.Sequential([

    keras.layers.Dropout(0.3, input_shape=train_vector.shape[1:]),

    keras.layers.Dense(64, activation= 'relu', kernel_initializer= 'he_normal', ),

    keras.layers.Dropout(0.1),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp.compile(optimizer= 'adam',

                  loss= 'mse',

                  metrics= ['accuracy', f1]

                 )



# Train MLP

mlp.fit(train_vector.toarray(), y_train, epochs=20)



# Evaluate model in validation dataset

test_loss, test_acc, test_f1 = mlp.evaluate(val_vector.toarray(),y_val, verbose= 0)

print("MLP Accuracy on Validation: {}; F1 Score: {}".format(test_acc, test_f1))



# Evaluate model in validation dataset

y_pred = mlp.predict_classes(val_vector.toarray())

score = f1_score(y_val, y_pred)

print("F1 Score MLP and Count Vectorizer Select K Best on validation dataset: {}".format(score))



# With current architecture it gives better performance f1_score at 0.760
# Train model on all training data

# Test using test dataset



# Create vectorizer and fit with training data

vectorizer = CountVectorizer(ngram_range= (1,2), max_features= 4000)

vectorizer.fit(train['text'])



# Create vector

train_vector = vectorizer.transform(train['text'])

test_vector = vectorizer.transform(test['text'])



# Select K best

selector = SelectKBest(f_classif, k = 3000)

selector.fit(train_vector, train.target)



# Select vector

train_vector = selector.transform(train_vector)

test_vector = selector.transform(test_vector)



# MLP Architecture

mlp = keras.Sequential([

    keras.layers.Dropout(0.3, input_shape=train_vector.shape[1:]),

    keras.layers.Dense(64, activation= 'relu', kernel_initializer= 'he_normal', ),

    keras.layers.Dropout(0.1),

    keras.layers.Dense(1, activation='sigmoid')

])



# Compile MLP

mlp.compile(optimizer= 'adam',

                  loss= 'mse',

                  metrics= ['accuracy']

                 )



# Train MLP

mlp.fit(train_vector.toarray(), train.target, epochs=20, verbose=0)



# Prediction on test dataset

y_pred = mlp.predict_classes(test_vector.toarray())



# Submission using MLP and count vectorizer

submission_mlp_select = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission_mlp_select['target'] = y_pred

submission_mlp_select.to_csv('count_select_mlp.csv', index=False)