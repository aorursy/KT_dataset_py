import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def plot_words_polarity(tfidf):
    lr = LogisticRegression()
    p = make_pipeline(tfidf, lr)
    p.fit(df['comment_text'].values, df['toxic'].values)

    rev = sorted({v: k for k,v in p.steps[0][1].vocabulary_.items()}.items())
    polarity = pd.DataFrame({'coef': p.steps[1][1].coef_[0]}, 
                            index = [i[1] for i in rev]).sort_values('coef')

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(1,2,1)
    polarity.tail(40).plot(kind='barh', color='red', ax=ax)
    ax = plt.subplot(1,2,2)
    polarity.head(40).plot(kind='barh', color='green', ax=ax)
df = pd.read_csv('../input/train.csv')
tfidf = TfidfVectorizer(lowercase=True, max_features=50000)
plot_words_polarity(tfidf)
tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(2,2))
plot_words_polarity(tfidf)
tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(3,3))
plot_words_polarity(tfidf)
tfidf = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1,1))
plot_words_polarity(tfidf)