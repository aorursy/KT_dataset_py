
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, plot_confusion_matrix
#for oversampling minority class
from imblearn.over_sampling import SMOTE

data = pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
data.head()
sample = data.sample(5)
for i in range(5):
    print('Class: ', sample.iloc[i]['label'])
    print('Email:')
    print(sample.iloc[i]['text'])
    print('\n', '---'*45)
    
#drop the ID column
data.drop('Unnamed: 0', axis = 1, inplace = True)
print(data['label_num'].value_counts()/sum(data['label_num'].value_counts())*100)
sns.countplot('label_num', data = data)
stopwords_set = set(stopwords.words('english'))
#Save the 'not'
#stopwords_set.remove('not')
#add subject to stopwords
stopwords_set.add('subject')
stopwords_set.add('http')
def preprocessing_text(x):
    import string
    #lower case
    x = x.lower()
    #remove number
    x = re.sub(r'\d+','',x)
    #remove punctuation
    x = re.sub(r'[^\w\s]', '',x)
    #remove leading and ending space
    x = x.strip()
    #remove stopword
    x = ' '.join([word for word in word_tokenize(x) if not word in stopwords_set])
    return x
#apply preprocessing text on text
data['text'] = data['text'].apply(lambda x: preprocessing_text(x))
train, test = train_test_split(data, test_size = 0.2, random_state = 42)
#draw wordcloud
from wordcloud import WordCloud
sns.set(style = None)
train_spam = train[train['label_num'] == 1]
train_spam = train_spam['text']
#turn series to string by join ' ' to it
train_spam = ' '.join(train_spam)
train_ham = train[train['label_num'] == 0]
train_ham = train_ham['text']
train_ham = ' '.join(train_ham)
wordcloud_spam = WordCloud(background_color = 'black', width = 2500, height = 2000 ).generate(train_spam)
plt.figure(figsize = (13,13))
print('Spam email wordcloud')
plt.imshow(wordcloud_spam)
plt.show()
wordcloud_ham = WordCloud(background_color = 'white', width = 2500, height = 2000).generate(train_ham)
print('Ham email wordcloud')
plt.figure(figsize = (13,13))

plt.imshow(wordcloud_ham)
plt.show()

train.iloc[0]['text']
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#function take in a list of tokenized word and stemming it 
def stemming_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return ' '.join(stemmed_words)
stemmer = PorterStemmer()
#tokenize word before use stemming
train['text'] = train['text'].apply(lambda x: word_tokenize(x))
train['text'] = train['text'].apply(lambda x: stemming_words(x))
train.iloc[0]['text']
#control the max_features to vectorize to leave some unpopular word out, which is consider not important, like personal names, ...
tfidf = TfidfVectorizer( strip_accents = 'ascii', max_df = 0.8, max_features = 27000)
train_vectorized = tfidf.fit_transform(train['text'])
test_vectorized = tfidf.transform(test['text'])
sm = SMOTE(sampling_strategy = 1,random_state = 42)
X_resample,y_resample = sm.fit_resample(train_vectorized, train['label_num'])
print(y_resample.value_counts()/sum(y_resample.value_counts())*100)
sns.countplot(y_resample)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix
nb = MultinomialNB()
nb.fit(train_vectorized, train['label_num'])
p = nb.predict(test_vectorized)
print('Naive Bayes on non-resample dataset\n\n')
print(classification_report(test['label_num'],p))
plot_confusion_matrix(nb, test_vectorized, test['label_num'], cmap = 'Paired')

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix
nb = MultinomialNB()
nb.fit(X_resample, y_resample)
p = nb.predict(test_vectorized)
print('Naive Bayes on resample dataset\n\n')
print(classification_report(test['label_num'],p))
plot_confusion_matrix(nb, test_vectorized, test['label_num'], cmap = 'Paired')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_vectorized, train['label_num'])
lr_p = lr.predict(test_vectorized)
print('Logistic regression on non-resample dataset')

print(classification_report(test['label_num'], lr_p))
plot_confusion_matrix(lr, test_vectorized, test['label_num'], cmap = 'Paired')
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_resample, y_resample)
lr_p = lr.predict(test_vectorized)
print('Logistic regression on resample dataset\n\n')
print(classification_report(test['label_num'], lr_p))
plot_confusion_matrix(lr, test_vectorized, test['label_num'], cmap = 'Blues')

from sklearn.svm import SVC
svc = SVC()
svc.fit(train_vectorized, train['label_num'])
svc_p = svc.predict(test_vectorized)
print('SVC on non-resample dataset')
print(classification_report(test['label_num'], svc_p))
plot_confusion_matrix(svc, test_vectorized, test['label_num'])
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_resample, y_resample)
svc_p = svc.predict(test_vectorized)
print(classification_report(test['label_num'], svc_p))
plot_confusion_matrix(svc, test_vectorized, test['label_num'])
#split the data again
train, test = train_test_split(data, test_size = 0.2, random_state = 42)
print('Email before lemmazation')
train.iloc[0]['text']
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#lemmatize function take in list of words, so you have to tokenize word before give it to this function
def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(lemmatizer.lemmatize(word))
    return ' '.join(lemmatized_words)
train.loc[:,'text'] = train.loc[:,'text'].apply(lambda x: word_tokenize(x))
train.loc[:,'text'] = train.loc[:,'text'].apply(lambda x: lemmatize_words(x))
print('Email after lemmazation')
train.iloc[0]['text']
#Vectorize text 
tfidf = TfidfVectorizer( strip_accents = 'ascii', max_df = 0.8, max_features = 27000)
train_vectorized = tfidf.fit_transform(train['text'])
test_vectorized = tfidf.transform(test['text'])
#and oversampling data 
sm = SMOTE(sampling_strategy = 1,random_state = 42)
X_resample,y_resample = sm.fit_resample(train_vectorized, train['label_num'])

nb = MultinomialNB()
nb.fit(train_vectorized, train['label_num'])
p = nb.predict(test_vectorized)
print('Naive bayes on non-resample dataset')
print(classification_report(test['label_num'],p))
plot_confusion_matrix(nb, test_vectorized, test['label_num'], cmap = 'Paired')

nb = MultinomialNB()
nb.fit(X_resample, y_resample)
p = nb.predict(test_vectorized)
print('Naive bayes on resample dataset')
print(classification_report(test['label_num'],p))
plot_confusion_matrix(nb, test_vectorized, test['label_num'], cmap = 'Paired')

lr = LogisticRegression()
lr.fit(train_vectorized, train['label_num'])
lr_p = lr.predict(test_vectorized)
print('Logistic regression on non-resample data')
print(classification_report(test['label_num'], lr_p))
plot_confusion_matrix(lr, test_vectorized, test['label_num'], cmap = 'Paired')
lr = LogisticRegression()
lr.fit(X_resample, y_resample)
lr_p = lr.predict(test_vectorized)
print('Logistic regression on resample data')
print(classification_report(test['label_num'], lr_p))
plot_confusion_matrix(lr, test_vectorized, test['label_num'], cmap = 'Blues')

svc = SVC()
svc.fit(train_vectorized, train['label_num'])
svc_p = svc.predict(test_vectorized)
print('SVC on non-resample data\n\n')
print(classification_report(test['label_num'], svc_p))
plot_confusion_matrix(svc, test_vectorized, test['label_num'], cmap = 'Blues')

svc = SVC()
svc.fit(X_resample, y_resample)
svc_p = svc.predict(test_vectorized)
print('SVC on resample data\n\n')
print(classification_report(test['label_num'], svc_p))
plot_confusion_matrix(svc, test_vectorized, test['label_num'], cmap = 'Blues')
