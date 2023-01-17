_MOST_COMMON_COUNT = 500
import pandas as pd

from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import wordpunct_tokenize

from nltk.corpus import stopwords

import unicodedata

from collections import Counter

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/spam.csv', encoding='Windows-1252')

df = df.loc[:, ['v1', 'v2']]

df = df.dropna()

df.head()
stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer("english")

def normalize_text(text):

    tokens = wordpunct_tokenize(text)

    tokens = [

        unicodedata.normalize('NFD', stemmer.stem(t) )

        for t in tokens 

        if t not in stop_words

    ]

    sentence = ' '.join(tokens)

    return sentence
# Updating the table column using the function defined above

df.loc[:, 'v2'] = df.loc[:, 'v2'].apply(normalize_text)
counter = Counter()

for line in df.loc[:, 'v2']:

    counter.update(wordpunct_tokenize(line))
most_common = [w[0] for w in counter.most_common(_MOST_COMMON_COUNT)]

word_map = {str(word): index for (word, index) in zip(most_common, range(len(most_common)))}
def transform_to_freq(text):

    features = np.zeros(_MOST_COMMON_COUNT)

    for word in wordpunct_tokenize(text):

        if str(word) in word_map:

            word_index = word_map[str(word)]

            features[word_index] = features[word_index] + 1

    return features.tolist()
# Declares the features and labels matrixes

X = np.array([transform_to_freq(line) for line in df.loc[:, 'v2']])

y = (df.loc[:, 'v1'].values == 'spam').reshape(-1, 1)
# Splits the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fits the model with the train split and generates predictions based on test data

regressor = LogisticRegression(solver='lbfgs').fit(X_train, y_train.ravel())

predictions = regressor.predict(X_test)
# Confusion matrix, used to extract metrics below

cm = confusion_matrix(y_test, predictions)

cm
tn = cm[0][0] # True Negatives

fn = cm[1][0] # False Negatives

tp = cm[1][1] # True Positives

fp = cm[0][1] # False Positives



recall = tp / (tp + fn)

precision = tp / (tp + fp)

accuracy = (tp + tn) / (tn + fn + tp + fp)

f1_score = 2 * ((precision * recall) / (precision + recall))

f1_score