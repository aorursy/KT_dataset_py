import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud
import os

os.listdir("../input/movie-review-sentiment-analysis-kernels-only/")
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip', sep="\t")

test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip', sep="\t")



# train = pd.read_csv("train.tsv.zip", compression='zip', index_col='PhraseId')

# test = pd.read_csv("test.tsv.zip", compression='zip', index_col ='PhraseId')
train
print(len(train['SentenceId'].unique()))

print(len(train['Phrase'].unique()))
print(len(test['SentenceId'].unique()))

print(len(test['Phrase'].unique()))
train.loc[train.SentenceId == 1]
train.loc[train.SentenceId == 1][train.Sentiment != 2]
# plt.hist(train['Sentiment'])

sns.countplot(train['Sentiment'])
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
def clean_text(words):

    review_corpus = []

    for i in range(0,len(words)):

        review = str(words[i])

        stop_words = set(stopwords.words('english'))

#          review  = [w for w in review if not w in stop_words]

        porter = PorterStemmer()

        review = [porter.stem(w) for w in word_tokenize(review.lower()) if w.isalpha() and not w in stop_words]

#         review = [porter.stem(word) for word in review]

        review = ' '.join(review)

#         review=' '.join([porter.stem(word) for word in review])

        review_corpus.append(review)

    return review_corpus

#     return review

#         word = word_tokenize(w)

#     return word    
train['clean_review']=clean_text(train.Phrase.values)

test['clean_review'] = clean_text(test.Phrase.values)
text_pos = train['clean_review'].loc[train.Sentiment == 4]
text_neg = train['clean_review'].loc[train.Sentiment == 0]
wordcloud = WordCloud(max_font_size=50, max_words=10, background_color="white").generate(str(text_pos))

plt.imshow(wordcloud)
wordcloud = WordCloud(max_font_size=50, max_words=10, background_color="white").generate(str(text_neg))

plt.imshow(wordcloud)




vectorizer = CountVectorizer()

vectorizer.fit(text_pos)

p = list(vectorizer.vocabulary_)

print(p[0:10])



vectorizer.fit(text_neg)

n = list(vectorizer.vocabulary_)

print(n[0:10])

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score


final = train['clean_review']

target = train['Sentiment']



final_test = test['clean_review']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
vectorizer = CountVectorizer(ngram_range = (1,2))

# vectorizer = TfidfVectorizer()

vectorizer.fit(final.values)

vector_final = vectorizer.transform(final.values)

# vector_target = vectorizer.transform(target.values)



vector_test = vectorizer.transform(final_test.values)
lr = LogisticRegression(penalty='l2', max_iter=1000)

lr.fit(vector_final, target)

predictions = lr.predict(vector_test)



# clf = MultinomialNB()

# clf.fit(vector_final, target)

# predictions = clf.predict(vector_test)
samplesub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')



output = pd.DataFrame({'PhraseId': samplesub.PhraseId, 'Sentiment': predictions})

output.to_csv('submission.csv', index=False)