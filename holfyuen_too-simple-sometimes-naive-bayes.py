# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize, FreqDist
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
resources = pd.read_csv("../input/resources.csv")
train.head()
train.loc[:,['project_essay_3','project_essay_4']] = train.loc[:,['project_essay_3','project_essay_4']].fillna('None')
test.loc[:,['project_essay_3','project_essay_4']] = test.loc[:,['project_essay_3','project_essay_4']].fillna('None')
essays_train = train[['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']].apply(lambda x: ' '.join(x), axis=1)
essays_test = test[['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']].apply(lambda x: ' '.join(x), axis=1)
essays_train = essays_train.str.replace(r'\\[rn]', '')
essays_test = essays_test.str.replace(r'\\[rn]', '')
essays_train = essays_train.str.replace('[{}]'.format(string.punctuation.replace('-','')), '') # Remove punctuation, keep hyphen
essays_test = essays_test.str.replace('[{}]'.format(string.punctuation.replace('-','')), '')
wnl = WordNetLemmatizer()

def lemmatize_text(text):
    return [wnl.lemmatize(wnl.lemmatize(w), pos='v') for w in word_tokenize(text)]

# essay_token = essays_train[:100].apply(lambda x: lemmatize_text(x))
# essay_token[0]
plt.figure(figsize=(15,10))
stopwords = set(STOPWORDS)

plt.subplot(121)
words_acc = train.loc[train['project_is_approved']==1,'project_title']
word_freq_acc = FreqDist(w for w in word_tokenize(' '.join(words_acc).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_freq_acc)
plt.imshow(wordcloud)
plt.title("Wordcloud (Title) - Approved", fontsize=12)
plt.axis("off")

plt.subplot(122)
words_rej = train.loc[train['project_is_approved']==0,'project_title']
word_freq_rej = FreqDist(w for w in word_tokenize(' '.join(words_rej).lower()) if (w not in stopwords) & (w.isalpha()))
wordcloud = WordCloud(background_color = 'white', height=300, max_words=100).generate_from_frequencies(word_freq_rej)
plt.imshow(wordcloud)
plt.title("Wordcloud (Title) - Rejected", fontsize=12)
plt.axis("off")
plt.show()
separate = False

if separate:
    count_vec = CountVectorizer(stop_words='english', max_features=10000, max_df=0.999, lowercase=True)
    x_train_e = count_vec.fit_transform(essays_train)
    x_test_e = count_vec.transform(essays_test)
    count_vec2 = CountVectorizer(stop_words='english', max_features=5000, max_df=0.999, lowercase=True)
    x_train_t = count_vec2.fit_transform(train['project_title'])
    x_test_t = count_vec2.transform(test['project_title'])
    x_train = hstack([x_train_e, x_train_t])
    x_test = hstack([x_test_e, x_test_t])
else:
    count_vec = CountVectorizer(stop_words='english', max_features=10000, max_df=0.999, lowercase=True)
    x_train = count_vec.fit_transform(essays_train + ' ' + train['project_title'])
    x_test = count_vec.transform(essays_test + ' ' + test['project_title'])
y_train = train['project_is_approved']
# split into train and validation set
xtr, xv, ytr, yv = train_test_split(x_train, y_train, test_size=0.2, random_state=6894)
nb = MultinomialNB()
nb.fit(xtr, ytr)
pred0 = nb.predict_proba(xv)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(yv, pred0[:,1], pos_label=1)
metrics.auc(fpr, tpr)
nb = MultinomialNB()
nb.fit(x_train, y_train)
pred1 = nb.predict_proba(x_test)
subnb = pd.concat([sub, pd.Series(pred1[:,1], name='project_is_approved')], axis=1).iloc[:,[0,2]]
subnb.to_csv('subnb.csv', index=False)
by_grade = train.groupby('project_grade_category', as_index=False)['project_is_approved'].mean()
by_cat = train.groupby('project_subject_categories', as_index=False)['project_is_approved'].mean()
by_subcat = train.groupby('project_subject_subcategories', as_index=False)['project_is_approved'].mean()
train_1 = pd.merge(train, by_grade, how='left', on='project_grade_category', suffixes = ['','_grade'])
train_1 = pd.merge(train_1, by_cat, how='left', on='project_subject_categories', suffixes = ['','_cat'])
train_1 = pd.merge(train_1, by_subcat, how='left', on='project_subject_subcategories', suffixes = ['','_subcat'])
train_1.head()
test_1 = pd.merge(test, by_grade, how='left', on='project_grade_category')
test_1 = pd.merge(test_1, by_cat, how='left', on='project_subject_categories', suffixes = ['','_cat'])
test_1 = pd.merge(test_1, by_subcat, how='left', on='project_subject_subcategories', suffixes = ['','_subcat'])
test_1 = test_1.rename(index=str, columns={'project_is_approved':'project_is_approved_grade'})
test_1.head()
test_1['project_is_approved_subcat'].fillna(np.mean(train.project_is_approved), inplace=True)
test_1[['project_is_approved_grade','project_is_approved_cat','project_is_approved_subcat']].isnull().sum()
lr = LogisticRegression()
x_train_lr = train_1[['project_is_approved_grade','project_is_approved_cat','project_is_approved_subcat']]
x_test_lr = test_1[['project_is_approved_grade','project_is_approved_cat','project_is_approved_subcat']]
lr.fit(x_train_lr, y_train)
pred2 = lr.predict_proba(x_test_lr)
sublr = pd.concat([sub, pd.Series(pred2[:,1], name='project_is_approved')], axis=1).iloc[:,[0,2]]
sublr.to_csv('sublr.csv', index=False)
pred3 = pred1[:,1]*0.7 + pred2[:,1]*0.3
suben = pd.concat([sub, pd.Series(pred3, name='project_is_approved')], axis=1).iloc[:,[0,2]]
suben.to_csv('suben.csv', index=False)