import pandas as pd

import matplotlib.pyplot as plt



%pylab inline

pylab.rcParams['figure.figsize'] = (10, 7)



df_fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

df_true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')



df_fake.date = pd.to_datetime(df_fake.date, errors='coerce')  # Some dates are links to article

df_true.date = pd.to_datetime(df_true.date)
print('Some fake news:')

for title, content in zip(df_fake.title[:5], df_fake.text[:5]):

    print(f'Title: {title}')

    print(f'Content: {content[:100]}...\n')

    

print('\n\nSome true news:')

for title, content in zip(df_true.title[:5], df_true.text[:5]):

    print(f'Title: {title}')

    print(f'Content: {content[:100]}...\n')
num_fake = len(df_fake)

num_true = len(df_true)

print(f'Total fake news: {num_fake:,}')

print(f'Total real news: {num_true:,}')



min_date = min(df_fake.date.min(), df_true.date.min())

max_date = max(df_fake.date.max(), df_true.date.max())

print(f'News published between {min_date} and {max_date}.')
import math  # math.isnan

from collections import defaultdict



# Fake news count

date_count = defaultdict(int)

for date in df_fake.date:

    date_count[(date.year, date.month//4)] += 1  # Group months by 4



X, Y = [], []

for year, month in sorted(date_count.keys()):

    if math.isnan(year) or math.isnan(month):

        continue  # Some dates are invalid

        

    X.append(f'{year} - {month+1}')

    Y.append(date_count[(year, month)])



plt.bar(X, Y, label='Fake news', alpha=0.8)



# Real news count

date_count = defaultdict(int)

for date in df_true.date:

    date_count[(date.year, date.month//4)] += 1  # Group months by 4



X, Y = [], []

for year, month in sorted(date_count.keys()):

    if math.isnan(year) or math.isnan(month):

        continue  # Some dates are invalid

        

    X.append(f'{year} - {month+1}')

    Y.append(date_count[(year, month)])



plt.bar(X, Y, label='Real news', alpha=0.7)



plt.title('Total news over time')

plt.xticks(rotation='45')

plt.legend()

plt.show()
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(min_df=100, stop_words='english')

pylab.rcParams['figure.figsize'] = (20, 14)

pylab.rcParams['font.size'] = 14



# Fake news

vect.fit(df_fake.title)

wc = WordCloud(width=1920, height=1080).generate_from_frequencies(vect.vocabulary_)



plt.subplot(2, 1, 1)

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.title('Most frequent words in fake news titles')



# Real news

vect.fit(df_true.title)

wc = WordCloud(width=1920, height=1080).generate_from_frequencies(vect.vocabulary_)



plt.subplot(2, 1, 2)

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.title('Most frequent words in true news titles')



plt.show()

pylab.rcParams['figure.figsize'] = (10, 7)  # Default parameters

pylab.rcParams['font.size'] = 10
import random



# Create train and test datasets

random.seed(0)



X = list(df_fake.title) + list(df_true.title)

y = [1 for _ in range(len(df_fake))] + [0 for _ in range(len(df_true))]



index = [i for i in range(len(X))]

random.shuffle(index)

X, y = [X[i] for i in index], [y[i] for i in index] # Shuffle both lists adequatly



prct_train = 0.8

cut = int(len(X) * prct_train)

X_train, y_train = X[:cut], y[:cut]

X_test, y_test = X[cut:], y[cut:]



print(f'Training on {len(X_train):,} exemples ({int(100*prct_train)}% of the dataset).')
# Eval functions



def accuracy(y_pred, y_test):

    score = [1 if pred == test else 0 for pred, test in zip(y_pred, y_test)]

    return sum(score) / len(y_pred)



def recall(y_pred, y_test):

    score = [1 if pred == 1 and test == 1 else 0 for pred, test in zip(y_pred, y_test)]

    return sum(score) / sum(y_test)



def F1(y_pred, y_test):

    acc, rec = accuracy(y_pred, y_test), recall(y_pred, y_test)

    return 2 * acc * rec / (acc + rec)



def print_scores(y_pred, y_test):

    acc = accuracy(y_pred, y_test)

    rec = recall(y_pred, y_test)

    f1 = F1(y_pred, y_test)

    

    print('Accuracy\tRecall\t\tF1')

    print(f'{round(acc, 2)}\t\t{round(rec, 2)}\t\t{round(f1, 2)}')
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline



pipe = Pipeline([('count', CountVectorizer()),

                 ('tfid', TfidfTransformer())]).fit(X_train)



count_train = pipe['count'].transform(X_train)

count_test = pipe['count'].transform(X_test)



tfidf_train = pipe.transform(X_train)

tfidf_test = pipe.transform(X_test)



print(f'Vocabulary size: {tfidf_train.shape[1]}')

print(f'Matrix shape: {tfidf_train.shape}')
from sklearn.naive_bayes import MultinomialNB



# CountVectorizer

classifier = MultinomialNB()

classifier.fit(count_train, y_train)



y_pred = classifier.predict(count_test)

print('CountVectorizer scores:')

print_scores(y_pred, y_test)



# Tf-idf

classifier = MultinomialNB()

classifier.fit(tfidf_train, y_train)



y_pred = classifier.predict(tfidf_test)

print('\nTf-idf scores:')

print_scores(y_pred, y_test)
probs = classifier.feature_log_prob_

prob_true = np.exp(probs[0, :])  # True news features prob

prob_fake = np.exp(probs[1, :])  # Fake news features prob

features = pipe['count'].get_feature_names()



pylab.rcParams['figure.figsize'] = (20, 14)

pylab.rcParams['font.size'] = 14





true_features = {word: prob for prob, word in zip(prob_true, features)}

wc = WordCloud(width=1920, height=1080).generate_from_frequencies(true_features)



plt.subplot(2, 1, 1)

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.title('Most impactful words for true news prediction')



fake_features = {word: prob for prob, word in zip(prob_fake, features)}

wc = WordCloud(width=1920, height=1080).generate_from_frequencies(fake_features)



plt.subplot(2, 1, 2)

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.title('Most impactful words for fake news prediction')

pylab.rcParams['figure.figsize'] = (10, 7)  # Default parameters

pylab.rcParams['font.size'] = 10
X = list(df_fake.text) + list(df_true.text)

y = [1 for _ in range(len(df_fake))] + [0 for _ in range(len(df_true))]



index = [i for i in range(len(X))]

random.shuffle(index)

X, y = [X[i] for i in index], [y[i] for i in index] # Shuffle both lists adequatly



prct_train = 0.8

cut = int(len(X) * prct_train)

X_train, y_train = X[:cut], y[:cut]

X_test, y_test = X[cut:], y[cut:]



print(f'Training on {len(X_train):,} exemples ({int(100*prct_train)}% of the dataset).')
pipe = Pipeline([('count', CountVectorizer()),

                 ('tfid', TfidfTransformer())]).fit(X_train)



count_train = pipe['count'].transform(X_train)

count_test = pipe['count'].transform(X_test)



tfidf_train = pipe.transform(X_train)

tfidf_test = pipe.transform(X_test)



print(f'Vocabulary size: {tfidf_train.shape[1]:,}')

print(f'Matrix shape: {tfidf_train.shape}')
from sklearn.naive_bayes import MultinomialNB



# CountVectorizer

classifier = MultinomialNB()

classifier.fit(count_train, y_train)



y_pred = classifier.predict(count_test)

print('CountVectorizer scores:')

print_scores(y_pred, y_test)



# Tf-idf

classifier = MultinomialNB()

classifier.fit(tfidf_train, y_train)



y_pred = classifier.predict(tfidf_test)

print('\nTf-idf scores:')

print_scores(y_pred, y_test)