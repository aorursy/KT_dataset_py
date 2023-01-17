import numpy as np

import pandas as pd

import nltk

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB

# nltk.download('punkt')

# nltk.download('wordnet')

# nltk.download('stopwords')
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
train_df.head(10)
train_df.shape
train_df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

test_df.drop(['keyword', 'location'], axis=1, inplace=True)
train_df.head()
test_df.head()
# Lowercase words

def to_lower(row):

    return row.lower()



train_df['text'] = train_df['text'].apply(to_lower)

test_df['text'] = test_df['text'].apply(to_lower)
train_df['tokens'] = train_df['text'].apply(word_tokenize)

test_df['tokens'] = test_df['text'].apply(word_tokenize)
train_df['tokens'].iloc[5]
# Remove Stop Words and Punctuations

nltk_stop_words = list(stopwords.words('english'))

punctuations = list(string.punctuation)

stop_words = nltk_stop_words + punctuations



def remove_stop_words(row):

    return [w for w in row if not w in stop_words]



train_df['tokens_stops_removed'] = train_df['tokens'].apply(remove_stop_words)

test_df['tokens_stops_removed'] = test_df['tokens'].apply(remove_stop_words)

train_df['tokens_stops_removed'].iloc[5]
# Stemming words

# ps = PorterStemmer()



# def stem_words(row):

#     return [ps.stem(w) for w in row]



# train_df['tokens_stemmed'] = train_df['tokens_stops_removed'].apply(stem_words)

# test_df['tokens_stemmed'] = test_df['tokens_stops_removed'].apply(stem_words)
# Lemmatoze words

word_lem = WordNetLemmatizer()

    

def lemmatize_words(row):

    return [word_lem.lemmatize(w) for w in row]



train_df['tokens_lemmatized'] = train_df['tokens_stops_removed'].apply(lemmatize_words)

test_df['tokens_lemmatized'] = test_df['tokens_stops_removed'].apply(lemmatize_words)

print(train_df['tokens_lemmatized'].iloc[6])

print(train_df['tokens_lemmatized'].iloc[7])

print(train_df['tokens_lemmatized'].iloc[18])

print(train_df['tokens_lemmatized'].iloc[20])
### The text needs to be transformed to vectors so as the algorithms will be able make predictions ###

### TFIDF - how important a word is to a document in a collection of documents

#### Applying scikit-learn TfidfVectorizer on tokenized text - http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/





def dummy_fun(doc):

    return doc



tfidf = TfidfVectorizer(

    analyzer='word',

    tokenizer=dummy_fun,

    preprocessor=dummy_fun,

    token_pattern=None)



fitted_vectorizer = tfidf.fit(train_df['tokens_lemmatized'])

tfidf_vectorizer_vectors_train = fitted_vectorizer.transform(train_df['tokens_lemmatized'])



tfidf_vectorizer_vectors_test = fitted_vectorizer.transform(test_df['tokens_lemmatized'])
model = MultinomialNB()
model.fit(tfidf_vectorizer_vectors_train, train_df['target'])
predictions = model.predict(tfidf_vectorizer_vectors_test)
submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
filename = 'Predictions v1.csv'

submission.to_csv(filename,index=False)