# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

import itertools

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/a-fake-news-dataset-around-the-syrian-war/FA-KES-Dataset.csv',encoding='latin1')

df.head()
df.isnull().sum().sum()
print('There are {} rows and {} columns in train'.format(df.shape[0],df.shape[1]))
print(df.article_content.describe())
ddf = df[df.duplicated()]

print(ddf)
df.drop_duplicates(keep=False, inplace=True)
ddf = df[df.duplicated()]

print(ddf)
#Show Labels distribution



df['labels'].value_counts(normalize=True)

sns.countplot(x='labels', data=df)
df['source'].value_counts().plot(kind='barh')


df.groupby(['source','labels']).size().unstack().plot(kind='bar',stacked=False)

plt.figure(figsize=(20,10))

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

true_len=df[df['labels']==1]['article_content'].str.len()

ax1.hist(true_len,color='green')

ax1.set_title('Real News')

fake_len=df[df['labels']==0]['article_content'].str.len()

ax2.hist(fake_len,color='red')

ax2.set_title('Fake News')

fig.suptitle('Characters in an article')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

true_len=df[df['labels']==1]['article_content'].str.split().map(lambda x: len(x))

ax1.hist(true_len,color='green')

ax1.set_title('Real News')

fake_len=df[df['labels']==0]['article_content'].str.split().map(lambda x: len(x))

ax2.hist(fake_len,color='red')

ax2.set_title('Fake News')

fig.suptitle('Words in an article')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=df[df['labels']==1]['article_content'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')

ax1.set_title('Real')

word=df[df['labels']==0]['article_content'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')

ax2.set_title('Fake')

fig.suptitle('Average word length in each article')
mfreq = pd.Series(' '.join(df[df['labels']==1]['article_content']).split()).value_counts()[:25]

mfreq
vect = TfidfVectorizer(use_idf=True,max_df=0.40,min_df=0.1,stop_words='english').fit(df[df['labels']==1]['article_content'])

len(vect.get_feature_names())
list(vect.vocabulary_.keys())[:10]
true_tfidf=list(vect.vocabulary_.keys())

wordcloud = WordCloud(width=1600, height=800).generate(str(true_tfidf))

#  plot word cloud image.



plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
mfreq = pd.Series(' '.join(df[df['labels']==0]['article_content']).split()).value_counts()[:25]

mfreq
vect = TfidfVectorizer(use_idf=True,max_df=0.40,min_df=0.1,stop_words='english').fit(df[df['labels']==0]['article_content'])

len(vect.get_feature_names())
fake_tfidf=list(vect.vocabulary_.keys())

wordcloud = WordCloud(width=1600, height=800).generate(str(fake_tfidf))

#  plot word cloud image.



plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Intialize TfidfVectorizer

tfidf_vect=TfidfVectorizer(stop_words='english',max_df=0.4,min_df=0.1).fit(df['article_content'])

len(tfidf_vect.get_feature_names())
txt_tfidf=list(tfidf_vect.vocabulary_.keys())

wordcloud = WordCloud(width=1600, height=800).generate(str(txt_tfidf))

#  plot word cloud image.



plt.figure( figsize=(20,10), facecolor='k')

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='latin-1', ngram_range=(1, 2), stop_words='english')



features = tfidf.fit_transform(df.article_content).toarray()

labels = df.labels

features.shape
X_train, X_test, y_train, y_test = train_test_split(df['article_content'], df['labels'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



clf = MultinomialNB().fit(X_train_tfidf, y_train)
print(clf.predict(count_vect.transform(["The Syrian army has taken control of a strategic northwestern crossroads town, its latest gain in a weeks-long offensive against the country's last major rebel bastion."])))
models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0)]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

  model_name = model.__class__.__name__

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])





sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

plt.show()
cv_df.groupby('model_name').accuracy.mean()