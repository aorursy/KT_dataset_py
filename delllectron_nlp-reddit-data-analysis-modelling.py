import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

from wordcloud import WordCloud,STOPWORDS
from string import punctuation
from bs4 import BeautifulSoup
import re,string,unicodedata
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix

import keras
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking

df = pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
#show dataframe
df.head()
#show feature data types
df.info()
#show data basic stats by over_18 feature
df.groupby('over_18').describe()
#replace the boolean values on over_18 to numerical values
def replace_labels(x):
    if x == False:
        return 0
    else:
        return 1

df['over_18'] = df['over_18'].apply(replace_labels)
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), yticklabels=False);
df.isnull().sum()
df.drop(['id', 'author_flair_text', 'removed_by',
         'total_awards_received', 'awarders', 'created_utc', 'full_link'],
        axis=1, inplace=True)
#add title character length feature
df['title_length'] = df['title'].apply(lambda x: len(str(x)))
df = df.dropna()
print('DATAFRAME SHAPE: ',df.shape)
df.head()
df.isnull().sum()
plt.figure(figsize=(7,5))
plt.title('COUNTPLOT')
plt.xlabel('Class')
plt.ylabel('Count')
sns.barplot(x=['Under 18', 'Over 18'],y= df.over_18.value_counts(), palette='viridis');
df.over_18.value_counts()
fig, ax = plt.subplots(1,2, figsize=(14,5))
ax[0].set_title('UNDER_18')
ax[1].set_title('OVER_18')

sns.distplot(df[df['over_18']==0]['title_length'], ax=ax[0], color='steelblue');
sns.distplot(df[df['over_18']==1]['title_length'], ax=ax[1], color='salmon');
#drop data which title character lenght is less than 5
df.drop(df[df['title_length']<5].index, inplace =True)
# combine author and title
df['text'] = df['title'] + ' ' + df['author']
df.drop(['title', 'author'], axis=1, inplace=True)
df.over_18.value_counts()
SAMPLES = 10000

under_18 = df[df['over_18']==0]
under_18 = under_18[(under_18['title_length']>25) & (under_18['title_length']<75)].sample(frac=1)
under_18 = under_18[:SAMPLES]
over_18 = df[df['over_18']==1]

df_train = pd.concat([under_18, over_18])
df_train.index = np.arange(len(df_train))

#check data frame
print('SHAPE: ', df_train.shape)
df_train.head()

X = df_train.drop(['score', 'num_comments', 'over_18', 'title_length'], axis=1)
y = df_train['over_18']
plt.figure(figsize=(14,10))
wc_under18 = WordCloud(min_font_size=3, max_font_size=3000, 
                       width=1920, height=1080, 
                       stopwords=STOPWORDS).generate(str(''.join(df_train[df_train['over_18']==0]['text'])))

plt.imshow(wc_under18, interpolation='bilinear');
plt.figure(figsize=(14,10))
wc_over18 = WordCloud(min_font_size=3, max_font_size=3000, 
                       width=1920, height=1080, 
                       stopwords=STOPWORDS).generate(str(''.join(df_train[df_train['over_18']==1]['text'])))

plt.imshow(wc_over18, interpolation='bilinear');
topwords_over_18 = pd.Series(wc_over18.process_text(str(''.join(df_train[df_train['over_18']==1]['text'])))).sort_values(ascending=False)[:15]

plt.figure(figsize=(10,7))
sns.barplot(topwords_over_18.values, topwords_over_18.index, palette='magma');
topwords_under_18 = pd.Series(wc_over18.process_text(str(''.join(df_train[df_train['over_18']==0]['text'])))).sort_values(ascending=False)[:15]

plt.figure(figsize=(10,7))
sns.barplot(topwords_under_18.values, topwords_under_18.index, palette='coolwarm');
# get the stop words, punctuation, and also add the OC, deleted and OC deleted
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
removed_topwords = ['deleted', 'oc', 'deleted oc', 'oc deleted']
stop.update(removed_topwords)
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
#lematizing function
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return final_text
#process the text data
X.text = X.text.apply(lemmatize_words)
X.text = X.text.apply(lambda i: ' '.join(i))
#check data
X.text
#split the data
X_train, X_test, y_train, y_test = train_test_split(X.text, y, test_size=0.2, random_state=101)
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
#classification report
print(classification_report(y_test, predictions))
plt.figure(figsize=(5,5))
con_mat = confusion_matrix(y_test, predictions)

sns.heatmap(con_mat, annot=True, square=True);

plt.xlabel('Y_TRUE');
plt.ylabel('PREDICTIONS');
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
#classification report
print(classification_report(y_test, predictions))
plt.figure(figsize=(5,5))
con_mat = confusion_matrix(y_test, predictions)

sns.heatmap(con_mat, annot=True, square=True);

plt.xlabel('Y_TRUE');
plt.ylabel('PREDICTIONS');
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', LinearSVC())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
#classification report
print(classification_report(y_test, predictions))
plt.figure(figsize=(5,5))
con_mat = confusion_matrix(y_test, predictions)

sns.heatmap(con_mat, annot=True, square=True);

plt.xlabel('Y_TRUE');
plt.ylabel('PREDICTIONS');
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', XGBClassifier(loss = 'deviance',
                                    learning_rate = 0.02,
                                    n_estimators = 10,
                                    max_depth = 7,
                                    random_state=101))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
#classification report
print(classification_report(y_test, predictions))
plt.figure(figsize=(5,5))
con_mat = confusion_matrix(y_test, predictions)

sns.heatmap(con_mat, annot=True, square=True);

plt.xlabel('Y_TRUE');
plt.ylabel('PREDICTIONS');
pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', SGDClassifier(n_jobs=-1))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
#classification report
print(classification_report(y_test, predictions))
plt.figure(figsize=(5,5))
con_mat = confusion_matrix(y_test, predictions)

sns.heatmap(con_mat, annot=True, square=True);

plt.xlabel('Y_TRUE');
plt.ylabel('PREDICTIONS');