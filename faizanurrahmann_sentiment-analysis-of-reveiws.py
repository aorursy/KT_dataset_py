# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 50, 18

rcParams["axes.labelsize"] = 16

import seaborn as sns

# text processing library

import spacy

import re

from gensim import corpora, models, similarities

# model library

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# ignore warnings

import warnings

warnings.filterwarnings('ignore')
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# read data

data = pd.read_csv("../input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv")

data.head()
# print data columns

for columns in data.columns:

    print(columns)
# select relevent columns for sentiment of reviews

review_data = data[['Hotel_Name', 'Positive_Review', 'Negative_Review', 'Average_Score', 'Reviewer_Score']].copy()

review_data.head()
# Concatenating the positve review and negative review

review_data['reviews'] = review_data[['Positive_Review', 'Negative_Review']].apply(lambda x: ' '.join(x), axis = 1)

#review_data.head()

print(review_data.loc[1, 'reviews'])
# checking distribution of review score

print(f"Review score given by Customer: {review_data.Reviewer_Score.value_counts()}")

review_data.Reviewer_Score.value_counts().plot(kind='bar', title='Count of Reviews', figsize = (15, 4))
# Rounding the Review Score to nearest integer

review_data['round_review_score'] = review_data.Reviewer_Score.apply(lambda x: np.ceil(x))

review_data.round_review_score.value_counts().plot(kind = 'bar', figsize=(16, 8), title = 'distribution of reviews')
# Selecting subset of data for speedup the computation.

print(f"Before subseting, data size: {data.shape}")

reviews_df = review_data.sample(frac = 0.1, replace = False, random_state=42)

print(f"After subseting, data size: {reviews_df.shape}")

reviews_df.head()
# function to clean and lemmatize text and remove stopwords

from gensim.parsing.preprocessing import preprocess_string

from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric

from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short



CUSTOM_FILTERS = [lambda x: str(x), # encode in utf-8

                  lambda x: x.lower(), # convert to lowercase

                  # remove emails, urls etc

                  lambda x: re.sub('[^\s]*.com[^\s]*', "", x),

                  lambda x: re.sub('[^\s]*www.[^\s]*', "", x),

                  lambda x: re.sub('[^\s]*.co.uk[^\s]*', "", x),

                  # remove special charecter

                  lambda x: re.sub('[^\s]*[\*]+[^\s]*', "", x),

                  lambda x: re.sub(r'\([^)]*\)', '', x),

                  strip_tags, # remove html tags

                  strip_punctuation, # replace punctuation with space

                  strip_non_alphanum, # remove non-alphanumeric characters

                  strip_numeric, # remove numbers

                  remove_stopwords,# remove stopwords

                  strip_short, # remove words less than minsize=4 characters long

                  strip_multiple_whitespaces# remove repeating whitespaces

                 ]

nlp = spacy.load('en')



def text_preprocess(docs, logging=True):

    docs = [preprocess_string(text, CUSTOM_FILTERS) for text in docs]

    texts_out = []

    for doc in docs:

    # spacy processing-pipelines

        doc = nlp((" ".join(doc)),  # doc = text to tokenize => creates doc

                  # disable parts of the language processing pipeline we don't need here to speed up processing

                  disable=['ner', # named entity recognition

                           'tagger', # part-of-speech tagger

                           'textcat', # document label categorizer

                          ])

        texts_out.append([tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-'])

    return pd.Series(texts_out)



text_preprocess(reviews_df.reviews.iloc[10:15])
# apply text-preprocessing function to training set

%time train_corpus = text_preprocess(reviews_df.reviews)
# create ngrams

ngram_phraser_1 = models.Phrases(train_corpus, threshold=1)

ngram_phraser_2 = models.Phrases(train_corpus, threshold=10)

ngram_1 = models.phrases.Phraser(ngram_phraser_1)

ngram_2 = models.phrases.Phraser(ngram_phraser_2)

#print example

print(ngram_1[train_corpus[0]])

print(ngram_2[train_corpus[0]])
# apply n-gram model to corpus

texts_1 = [ngram_1[token] for token in train_corpus]

texts_2 = [ngram_2[token] for token in train_corpus]

# adding it to dataframe

texts_1 = [' '.join(text) for text in texts_1]

texts_2 = [' '.join(text) for text in texts_2]

reviews_df['ngram_1'] = texts_1

reviews_df['ngram_2'] = texts_2

reviews_df.head()
# visualizing relevent word in ngram_1

from wordcloud import WordCloud



text = ""

for i in range(reviews_df.shape[0]):

    text = " ".join([text,reviews_df["ngram_1"].values[i]])

    



wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=312)

plt.imshow(wordcloud)

plt.title("Wordcloud for reveiws ")

plt.axis("off")

plt.show()
# Check class distribution

fig = plt.figure(figsize=(15, 4))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

reviews_df.round_review_score.value_counts().plot(kind='bar', title='Before sampling Review distribution', ax = ax1)



# Dividing proportion of data by class

counts_category = reviews_df.round_review_score.value_counts().values



# Divide by class

df_class_10 = reviews_df[reviews_df.round_review_score == 10.0]

df_class_8 = reviews_df[reviews_df.round_review_score == 8.0]

df_class_9 = reviews_df[reviews_df.round_review_score == 9.0]

df_class_7 = reviews_df[reviews_df.round_review_score == 7.0]

df_class_5 = reviews_df[reviews_df.round_review_score == 5.0]

df_class_4 = reviews_df[reviews_df.round_review_score == 4.0]

df_class_3 = reviews_df[reviews_df.round_review_score == 3.0]



# random oversampling

df_class_10 = df_class_10.sample(counts_category[0], replace=False)

df_class_9 = df_class_9.sample(counts_category[0], replace=True)

df_class_8 = df_class_8.sample(counts_category[0], replace=True)

df_class_7 = df_class_7.sample(counts_category[0], replace=True)

df_class_5 = df_class_5.sample(counts_category[0], replace=True)

df_class_4 = df_class_4.sample(counts_category[0], replace=True)

df_class_3 = df_class_3.sample(counts_category[0], replace=True)



# concatenate individual datafram

df_train_oversampled = pd.concat([df_class_10, df_class_9, df_class_8, df_class_7, df_class_5, df_class_4, df_class_3], axis=0)



# Now, Check class distribution



df_train_oversampled.round_review_score.value_counts().plot(kind='bar', title='After sampling Review distribution', ax = ax2)

#df_train_oversampled.job_type.value_counts().plot(kind='bar', title='Count (job_type)', ax=ax2)
# import model library

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB

import xgboost as xgb

from sklearn.ensemble import VotingClassifier



# represent features in countvectorizer for ngram_1

vectorizer_1 = CountVectorizer()

vectorizer_1.fit(df_train_oversampled.ngram_1)



# split into test and train sets for ngram_1

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_train_oversampled.ngram_1, df_train_oversampled.round_review_score, test_size=0.4)



# represent features in countvectorizer for ngram_2

vectorizer_2 = CountVectorizer()

vectorizer_2.fit(df_train_oversampled.ngram_2)



# split into test and train sets for ngram_2

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_train_oversampled.ngram_2, df_train_oversampled.round_review_score, test_size=0.4)
# Build LogisticRegression Model

# for ngram_1

lr_1 = LogisticRegression()

lr_1.fit(vectorizer_1.transform(X_train_1), y_train_1)

# for ngram_2

lr_2 = LogisticRegression()

lr_2.fit(vectorizer_2.transform(X_train_2), y_train_2)



print('Logistic Regression Score on ngram_1 reviews: ', lr_1.score(vectorizer_1.transform(X_test_1), y_test_1))

print('Logistic Regression Score on ngram_2 reviews: ', lr_2.score(vectorizer_2.transform(X_test_2), y_test_2))



y_1 = lr_1.predict(vectorizer_1.transform(X_test_1))

print("classification report on ngram_1 reviews:\n ", classification_report(y_test_1, y_1))



y_2 = lr_2.predict(vectorizer_2.transform(X_test_2))

print("classification report on ngram_2 reviews:\n ", classification_report(y_test_2, y_2))

# create private test data from sample

test_data = reviews_df.sample(frac = 0.05, replace = False, random_state=42)

test_data = test_data[['reviews', 'ngram_1', 'ngram_2']].copy()

# pridiction on test data

#lr_1.fit(vectorizer_1.transform(df_train_oversampled.ngram_1), df_train_oversampled.round_review_score)

prediction_1 = lr_1.predict(vectorizer_1.transform(test_data.ngram_1))



#lr_2.fit(vectorizer_2.transform(df_train_oversampled.ngram_2), df_train_oversampled.round_review_score)

prediction_2 = lr_2.predict(vectorizer_2.transform(test_data.ngram_2))



sample_test_score = pd.DataFrame({'Review_ngram_1':test_data.ngram_1, 'Review_ngram_2': test_data.ngram_2, 'score_ngram_1':prediction_1, 'score_ngram_2':prediction_2})

sample_test_score.to_csv('sample_test_score.csv', index=False)

sample_test_score.head()
# checking how the two pridicted score is different.

from scipy import stats

# Paired ttest

filter_data = sample_test_score.dropna(subset=['score_ngram_1', 'score_ngram_2'])

ttest, pval = stats.ttest_ind(filter_data.score_ngram_1, filter_data.score_ngram_2)

if pval<0.5:

  print("scores is almost same:", ttest, pval)

else:

  print("scores is different: ", ttest, pav)
fig = plt.figure(figsize=(15, 4))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

sample_test_score.score_ngram_1.value_counts().plot(kind='bar', title='ngram_1 pridiction', ax = ax1)

sample_test_score.score_ngram_2.value_counts().plot(kind='bar', title='ngram_2 pridiction', ax = ax2)
