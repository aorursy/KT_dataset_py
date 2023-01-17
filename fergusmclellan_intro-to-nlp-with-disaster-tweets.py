import pandas as pd

import numpy as np

from sklearn import feature_extraction, linear_model, model_selection, preprocessing 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# The training data contains a label column called "target" which is 0 for tweets which are "not real"

# (not referring to a real disaster) and 1 for tweets which are "real" (refers to a real disaster)

df_train[['text','target']].head(20)
# The test data only includes the tweet text, and is not labelled

df_test.head()
# Check how many entries are in the training and test sets

print(df_train.info())

print(df_test.info())
# We can compare how many tweets are "real" and "not real" by looking 

# at the value counts in the "targets" column

df_train["target"].value_counts()
# average length of each tweet

df_train['text_length'] = df_train['text'].str.len()

df_train['text_length'].mean()
# number of characters in "real" tweets 

df_train[df_train["target"]==1]['text_length'].hist()
# number of characters in "Not real" tweets 

df_train[df_train["target"]==0]['text_length'].hist()
import re



example_sentence = "Round and round the Radical Road, the radical rascal ran."

example_lowercase = example_sentence.lower()

example_punctuation_removed = re.sub("[,.!?;-]", " ", example_lowercase)

example_tokenize = example_punctuation_removed.split()

print(example_tokenize)
# A more comprehensive punctuation list is available in the "string" module

import string

punctuation=string.punctuation

print(punctuation)
# We can build a dictionary of the words, and store the frequency of each word in the dictionary

word_frequency = dict()

for word in example_tokenize:

    if word not in word_frequency.keys():

        word_frequency[word] = 1

    else:

        word_frequency[word] +=1

        

print(word_frequency)
import spacy

nlp=spacy.load("en_core_web_sm")

doc = nlp(example_sentence)

from spacy.lang.en.stop_words import STOP_WORDS

stopwords = list(STOP_WORDS)



for token in doc:

    lowercase_word = token.lower_

    if lowercase_word not in stopwords and lowercase_word not in punctuation:

        print(lowercase_word)
# We can define a function to combine all of these steps to tokenize, remove punctuation, and remove the stopwords

def text_tokenize_and_cleanup(input_text):

    if len(input_text) > 1 and type(input_text) == str:

        doc = nlp(input_text)

        output_tokens = []

        for token in doc:

            lowercase_word = token.lower_

            if lowercase_word not in stopwords and lowercase_word not in punctuation:

                output_tokens.append(lowercase_word)

        return output_tokens
example_tokenized_with_spacy = text_tokenize_and_cleanup(example_sentence)

print(example_tokenized_with_spacy)
count_vectorizer = feature_extraction.text.CountVectorizer(tokenizer=text_tokenize_and_cleanup)

example_transformed = count_vectorizer.fit_transform(example_tokenized_with_spacy)
# However, each word in the count vectorizer is actually represented by a number

print(example_transformed)
# Here the first 2 words are "round" and "round". Therefore, "round" is represented by the number 4

# vocabulary_ can be used to view the dictionary which is used to identify which word is which number:

print(count_vectorizer.vocabulary_)
# The "get_feature_names()" method can be used to view just the words from the vocabulary, in the order

# in which they appear. This can be combined with the array output of the transformed count_vectorizer

# output to allow both to be viewed easily together

print(count_vectorizer.get_feature_names())

print(example_transformed.toarray())
# The TfidfVectorizer will work across more than one tweet, therefore, we need to have more than one sample sentence in a simple demo.

# We will take the first 5 tweets from the training dataset to see how this works.

sample_text = df_train['text'].head(5)

pd.set_option("max_colwidth", -1)

print(sample_text)
sample_tfidf_vectorizer=feature_extraction.text.TfidfVectorizer(tokenizer=text_tokenize_and_cleanup)

sample_tfidf_vector = sample_tfidf_vectorizer.fit_transform(sample_text)
# Examine the first vector in the TfidfVectorizer (which corresponds to the values for the first document)

first_vector_tfidfvectorizer=sample_tfidf_vector[0]



df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=sample_tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

df.sort_values(by=["tfidf"],ascending=False).head(10)
# Examine the last vector out (for the 5th document)

fourth_vector_tfidfvectorizer=sample_tfidf_vector[4]

df = pd.DataFrame(fourth_vector_tfidfvectorizer.T.todense(), index=sample_tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

df.sort_values(by=["tfidf"],ascending=False).head(10)
# Split the training data so that we can use it to train and validate our models

# We will use this to determine which model provides the best predictions, and 

# use the best model to create our final Kaggle submission file.

X_train, X_test, y_train, y_test = train_test_split(df_train["text"],df_train["target"], test_size = 0.3)
# Confirm that we do not have badly imbalanced "real" and "not real" data in training and test datasets

print(y_train.value_counts())

print(y_test.value_counts())
print(X_train.head())
print(y_train.head())
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score
# Accuracy with basic Logistic Regression, using TFIDF



logreg = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LogisticRegression()),

               ])

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Compare with Linear Support Vector Machine - SGDClassifier



sgd = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier()),

               ])

sgd.fit(X_train, y_train)



y_pred = sgd.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Using Naive Bayes - MultinomialNB



nb = Pipeline([('vect', CountVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB()),

              ])

nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Accuracy with Random Forest



rf = Pipeline([('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', RandomForestClassifier()),

               ])

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Accuracy with basic Logistic Regression, not using TFIDF



logreg = Pipeline([('vect', CountVectorizer()),

                ('clf', LogisticRegression()),

               ])

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Compare with Linear Support Vector Machine - SGDClassifier



sgd = Pipeline([('vect', CountVectorizer()),

                ('clf', SGDClassifier()),

               ])

sgd.fit(X_train, y_train)



y_pred = sgd.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Using Naive Bayes - MultinomialNB



nb = Pipeline([('vect', CountVectorizer()),

               ('clf', MultinomialNB()),

              ])

nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Accuracy with Random Forest



rf = Pipeline([('vect', CountVectorizer()),

                ('clf', RandomForestClassifier()),

               ])

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# This would indicate that the best classifier to use is MultinomialNB, with no TFIDF Transformer

# included in the Pipeline
# Let us consider the entire training set, and ignore splitting this into separate training and validation sets. 

all_training_text = df_train['text']



tfidf_vectorizer=feature_extraction.text.TfidfVectorizer()

X = tfidf_vectorizer.fit_transform(all_training_text)
print(X.shape)
test_sentence = "I am happy, but you are not."

doc = nlp(test_sentence)

for token in doc:

    print(token.text, token.lemma_)
# Define a function to lemmatize the dataframe (pronouns do not have a lemma_)

# Also, remove stopwords and punctuation

punct = string.punctuation

def lemmatize_text(input_docs, logging=False):

    text = []

    for doc in input_docs:

        doc = nlp(doc, disable=['parser', 'ner'])

        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punct]

        tokens = ' '.join(tokens)

        text.append(tokens)

    return pd.Series(text)
# test function against the training set

X_train_lemmas_dev = lemmatize_text(X_train)
print(X_train_lemmas_dev.head())
X_test_lemmas_dev = lemmatize_text(X_test)
# Using Naive Bayes - MultinomialNB



nb = Pipeline([('vect', CountVectorizer()),

                   ('clf', MultinomialNB()),

              ])

nb.fit(X_train_lemmas_dev, y_train)



y_pred = nb.predict(X_test_lemmas_dev)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# Accuracy with basic Logistic Regression



logreg = Pipeline([('vect', CountVectorizer()),

                ('clf', LogisticRegression()),

               ])

logreg.fit(X_train_lemmas_dev, y_train)



y_pred = logreg.predict(X_test_lemmas_dev)



print('accuracy %s' % accuracy_score(y_pred, y_test))
# prepare submission

X_train_lemmas = lemmatize_text(df_train["text"])

y_train = df_train["target"]

X_test_lemmas = lemmatize_text(df_test["text"])
nb.fit(X_train_lemmas, y_train,)

y_pred = nb.predict(X_test_lemmas)
df_test['target'] = np.rint(y_pred).astype(int)
print(df_test.head(20))
summary_output = df_test[['id','target']]

summary_output.to_csv('sample_submission.csv', index=False)