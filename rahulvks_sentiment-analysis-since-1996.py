import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import re
from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
eng_stopwords = set(stopwords.words("english"))
import matplotlib.gridspec as gridspec 
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.tsv',delimiter='\t')
test = pd.read_csv('../input/test.tsv',delimiter='\t')
sub = pd.read_csv('../input/sampleSubmission.csv')


print(train.shape, test.shape)

train.head()
train['Sentiment'].value_counts()
train.describe()
test.describe()

x=train['Sentiment'].value_counts()
#plot
plt.figure(figsize=(15,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
def cleaning(s):
    
    s = str(s)
    #s = str.split(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    #s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    s = re.sub(r'\<a href', ' ', s)
    s = re.sub(r'&amp;', '', s) 
    s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = re.sub(r'[^\x00-\x7f]',r'',s) #removes arabic
    s = re.sub(r'<br />', ' ', s)
    s = re.sub(r'\'', ' ', s)
    
    return s
train['Phrase_Clean'] = [cleaning(s) for s in train['Phrase']]
APPLY_STEMMING = True

if APPLY_STEMMING:
    import nltk.stem as stm # Import stem class from nltk
    stemmer = stm.PorterStemmer()
train.Phrase_Clean = train.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

train['count_word']=train["Phrase_Clean"].apply(lambda x: len(str(x).split()))
train['count_unique_word']=train["Phrase_Clean"].apply(lambda x: len(set(str(x).split())))
train["count_stopwords"] = train["Phrase_Clean"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["mean_word_len"] = train["Phrase_Clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

plt.figure(figsize=(12,6))
# words
plt.subplot(122)
sns.violinplot(y='count_word',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('Words Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

# words
#train['count_word'].loc[train['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_unique_word',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('Unique Word Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

# words
#train['count_word'].loc[train['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_stopwords',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('count stopwords Word Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()

plt.figure(figsize=(12,6))

# words
#train['count_word'].loc[train['count_word']>200] = 200
plt.subplot(122)
sns.violinplot(y='count_stopwords',x='Sentiment', data=train,split=True,inner="quart")
plt.xlabel('mean_word_len Vs Target', fontsize=12)
plt.ylabel('# of words', fontsize=12)
plt.title("Number of words in each comment", fontsize=15)

plt.show()
# build TFIDF Vectorizer
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,4),
    dtype=np.float32,
    max_features=20000
)


# Character Stemmer
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words = None,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 6),
    dtype=np.float32,
    max_features=30000
)

word_vectorizer.fit(train['Phrase_Clean'])

char_vectorizer.fit(train['Phrase_Clean'])
Target = train["Sentiment"]
# Train
train_word_features = word_vectorizer.transform(train['Phrase_Clean'])
train_char_features = char_vectorizer.transform(train['Phrase_Clean'])

train_features = hstack([
    train_char_features,
    train_word_features])

from sklearn.naive_bayes import MultinomialNB
model_NB = MultinomialNB()
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)
model_NB.fit(X_train_tfidf, y_train_tfidf)
predictions_tfidf = model_NB.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)
print(accuracy_tfidf)

print("Auc Score: ",np.mean(cross_val_score(model_NB, train_features, Target, cv=5,)))
print("Modeling..")
loss = []
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(train_features, Target, train_size=0.75)

lr = LogisticRegression(solver="liblinear", max_iter=500,class_weight='balanced')
lr.fit(train_features,Target)
lr_pred=lr.predict(X_test_tfidf)
accuracy_tfidf =accuracy_score(y_test_tfidf,lr_pred)
print(accuracy_tfidf)

print("Auc Score: ",np.mean(cross_val_score(lr, train_features, Target, cv=3,)))

Target_Names = train['Sentiment'].unique()
Target_Names
print(classification_report(y_test_tfidf,lr_pred))
test['Phrase_Clean'] = [cleaning(s) for s in test['Phrase']]
test.Phrase_Clean = test.Phrase_Clean.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))

# test
test_word_features = word_vectorizer.transform(test['Phrase_Clean'])
test_char_features = char_vectorizer.transform(test['Phrase_Clean'])
test_features = hstack([
    test_char_features,
    test_word_features])
predicted_values = lr.predict(test_features)
test.head()
test = test.drop('Phrase_Clean', 1)

test['Sentiment'] = predicted_values
test.head()
test[['PhraseId', 'Sentiment']].to_csv('submission_lr.csv', index=False)

test.shape

