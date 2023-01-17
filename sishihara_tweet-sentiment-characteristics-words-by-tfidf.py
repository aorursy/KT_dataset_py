import numpy as np

import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
# Because train.text contains NaN

# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138213

train.dropna(subset=['text'], inplace=True)
train.shape, test.shape
train.head()
positive_words = []

negative_words = []

neutral_words = []



for row in (train.query('sentiment=="positive"')['selected_text']):

    positive_words += row.split(' ')



for row in (train.query('sentiment=="negative"')['selected_text']):

    negative_words += row.split(' ')



for row in (train.query('sentiment=="neutral"')['selected_text']):

    neutral_words += row.split(' ')



data = [

    ' '.join(positive_words),

    ' '.join(negative_words),

    ' '.join(neutral_words)

]



stopWords = stopwords.words("english")

vectorizer = TfidfVectorizer(stop_words=stopWords)

X = vectorizer.fit_transform(data).toarray()



tfidf_df = pd.DataFrame(X.T, index=vectorizer.get_feature_names(),

                        columns=['positive', 'negative', 'neutral'])
tfidf_df.shape
tfidf_df.sort_values('positive', ascending=False).head(10)
tfidf_df.sort_values('negative', ascending=False).head(10)
tfidf_df.sort_values('neutral', ascending=False).head(10)