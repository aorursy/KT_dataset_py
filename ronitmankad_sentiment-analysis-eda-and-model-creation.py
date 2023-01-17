from sklearn import model_selection, preprocessing, linear_model, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score



import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



# Matplotlib forms basis for visualization in Python

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



# We will use the Seaborn library

import seaborn as sns

sns.set()



import os, string

# Graphics in SVG format are more sharp and legible

%config InlineBackend.figure_format = 'svg'

print(os.listdir('../input/'))
df = pd.read_csv('../input/train.csv')

df.head()
test_df = pd.read_csv('../input/test.csv')
df['sentiment'].value_counts()



sns.countplot(x='sentiment', data=df)

df['char_count'] = df['review'].apply(len)

df['word_count'] = df['review'].apply(lambda x: len(x.split()))

df['word_density'] = df['char_count'] / (df['word_count']+1)

df['punctuation_count'] = df['review'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df.head()
#df.hist(column=['char_count', 'word_count'])

features = ['char_count', 'word_count', 'word_density', 'punctuation_count']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharey=True)



for i, feature in enumerate(features):

    df.hist(column=feature, ax=axes.flatten()[i])

df[features].describe()
fig, ax = plt.subplots()

sns.boxplot(df['word_count'], order=range(0,max(df['word_count'])), ax=ax)



ax.xaxis.set_major_locator(ticker.MultipleLocator(200))

ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

fig.set_size_inches(8, 4)

plt.show()
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



X_train, X_val, y_train, y_val = train_test_split(df['review'], df['sentiment'], test_size=0.3,

random_state=17)
# Converting X_train and X_val to tfidf vectors (since out models can't take text data is input)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(df['review'])

xtrain_tfidf =  tfidf_vect.transform(X_train)

xvalid_tfidf =  tfidf_vect.transform(X_val)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(df['review'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(X_val)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(df['review'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_val) 



# Also creating for the X_test which is essentially test_df['review'] column

xtest_tfidf =  tfidf_vect.transform(test_df['review'])

xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_df['review'])

xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_df['review']) 
# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(df['review'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(X_train)

xvalid_count =  count_vect.transform(X_val)

xtest_count = count_vect.transform(test_df['review'])
model1 = linear_model.LogisticRegression()

model1.fit(xtrain_count, y_train)

accuracy=model1.score(xvalid_count, y_val)

print('Accuracy Count LR:', accuracy)

test_pred1=model1.predict(xtest_count)



model2 = linear_model.LogisticRegression()

model2.fit(xtrain_tfidf, y_train)

accuracy=model2.score(xvalid_tfidf, y_val)

print('Accuracy TFIDF LR:', accuracy)

test_pred2=model2.predict(xtest_tfidf)



model3 = linear_model.LogisticRegression()

model3.fit(xtrain_tfidf_ngram, y_train)

accuracy = model3.score(xvalid_tfidf_ngram, y_val)

print('Accuracy TFIDF NGRAM LR:', accuracy)

test_pred3 = model3.predict(xtest_tfidf_ngram)
final_pred = np.array([])

for i in range(0,len(test_df['review'])):

    final_pred = np.append(final_pred, np.argmax(np.bincount([test_pred1[i], test_pred2[i], test_pred3[i]])))
sub_df = pd.DataFrame()

sub_df['Id'] = test_df['Id']

sub_df['sentiment'] = [int(i) for i in final_pred]
sub_df.head()
sub_df.to_csv('my_submission.csv', index=False)