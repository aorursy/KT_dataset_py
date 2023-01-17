import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")
X_train_filepath = os.path.join('..', 'input', 'train.csv')
X_test_filepath = os.path.join('..', 'input', 'test.csv')
sample_filepath = os.path.join('..', 'input', 'sample_submission.csv')
X_train_filepath, X_test_filepath, sample_filepath
df_train = pd.read_csv(X_train_filepath, encoding='ISO-8859-1')
df_train.head()
df_test = pd.read_csv(X_test_filepath, encoding='ISO-8859-1')
df_test.head()
df_train.shape, df_test.shape
df_sample = pd.read_csv(sample_filepath, encoding='ISO-8859-1')
df_sample.head()
df_train.info()
df_train["target"].value_counts()
insincere = df_train[df_train["target"] == 1]
sincere = df_train[df_train["target"] == 0]
sincere.head()
sincere.info()
insincere.head()
insincere.info()
ax, fig = plt.subplots(figsize=(10, 7))
question_class = df_train["target"].value_counts()
question_class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()
print(df_train['target'].value_counts())
print(sum(df_train['target'] == 1) / sum(df_train['target'] == 0) * 100, "percent of questions are insincere.")
print(100 - sum(df_train['target'] == 1) / sum(df_train['target'] == 0) * 100, "percent of questions are sincere")
stop_words = stopwords.words("english")
insincere_words = ''

for question in insincere.question_text:
    text = question.lower()
    tokens = word_tokenize(text)
    for words in tokens:
        insincere_words = insincere_words + words + ' '
# Generate a word cloud image
insincere_wordcloud = WordCloud(width=600, height=400).generate(insincere_words)
#Insincere Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(insincere_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
insincere["questions_length"] = insincere.question_text.apply(lambda x: len(x))
sincere["questions_length"] = sincere.question_text.apply(lambda x: len(x))
insincere["questions_length"].mean()
sincere["questions_length"].mean()
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(insincere.questions_length, hist=True, label="insincere")
sns.distplot(sincere.questions_length, hist=True, label="sincere");
insincere['number_words'] = insincere.question_text.apply(lambda x: len(x.split()))
sincere['number_words'] = sincere.question_text.apply(lambda x: len(x.split()))
insincere['number_words'].mean()
sincere['number_words'].mean()
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(insincere.number_words, hist=True, label="insincere")
sns.distplot(sincere.number_words, hist=True, label="sincere");
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3),
                        strip_accents='unicode',lowercase =True, 
                        stop_words = 'english',tokenizer=word_tokenize)
train_vectorized = vectorizer.fit_transform(df_train.question_text.values)
test_vectorized = vectorizer.fit_transform(df_test.question_text.values)
X_train, X_val, y_train, y_val = train_test_split(train_vectorized, df_train.target.values, test_size=0.1, stratify = df_train.target.values)
lr = LogisticRegression(C=10, class_weight={0:0.07 , 1:1})
lr.fit(X_train, y_train)
y_pred_train1 = lr.predict(X_train)
print(f1_score(y_train, y_pred_train1))
y_pred_val1 = lr.predict(X_val)
print(f1_score(y_val, y_pred_val1))
cm1 = confusion_matrix(y_val, y_pred_val1)
cm1
sns.heatmap(cm1, cmap="Blues", annot=True, square=True, fmt=".0f");
print(classification_report(y_val, y_pred_val1))
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train, y_train)
y_pred_train2 = mnb.predict(X_train)
print(f1_score(y_train, y_pred_train2))
y_pred_val2 = mnb.predict(X_val)
print(f1_score(y_val, y_pred_val2))
cm2 = confusion_matrix(y_val, y_pred_val2)
cm2
sns.heatmap(cm2, cmap="Blues", annot=True, square=True, fmt=".0f");
print(classification_report(y_val, y_pred_val2))
from sklearn.svm import LinearSVC

svc = LinearSVC(C=5, class_weight={0:0.07 , 1:1})
svc.fit(X_train, y_train)
y_pred_train3 = svc.predict(X_train)
print(f1_score(y_train, y_pred_train3))
y_pred_val3 = svc.predict(X_val)
print(f1_score(y_val, y_pred_val3))
cm3 = confusion_matrix(y_val, y_pred_val3)
cm3
sns.heatmap(cm3, cmap="Blues", annot=True, square=True, fmt=".0f");
print(classification_report(y_val, y_pred_val3))