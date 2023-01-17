import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings 
import seaborn as sns
warnings.filterwarnings("ignore")

def clean_ColText(data, col, stem=True):
    """Takes dataframe and column name and returns a dataframe with cleaned strings in the form of a list. Stemming is an option."""
    df = data.copy()
    table = str.maketrans('', '', string.punctuation)
    df[col] = df[col].map(lambda x: x.translate(table)) #remove punctuation
    df[col] = df[col].map(lambda x: x.lower()) #lowercase
    df[col] = df[col].apply(word_tokenize) #tokenize
    stop_words = set(stopwords.words('english'))
    df[col] = df[col].map(lambda x: [y for y in x if not y in stop_words]) #remove stop words
    df[col] = df[col].map(lambda x: [y for y in x if y not in ["’","’","”","“","‘","—"]]) #remove smart quotes and other non alphanums
    if stem:
        porter = PorterStemmer()
        df[col] = df[col].map(lambda x: [porter.stem(y) for y in x])
        return df
    return df

pd.read_csv('../input/train.csv').head(10)
sid = SentimentIntensityAnalyzer()
df = pd.read_csv('../input/train.csv')
df.id = df.id.map(lambda x: int(x.replace('id','')))
df['sent'] = df['text'].map(lambda x: sid.polarity_scores(x)['compound'])
df = clean_ColText(df, 'text', stem=True)
df = df.drop('id', axis=1)
df.head(10)
plt.figure(figsize=(15,5))
plt.title('Sentiments of Authors')
sns.boxplot(x='author', y='sent', data=df)
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df['text'] = df.text.map(lambda x: ' '.join(x))
sent = np.array(df.sent) + 1
X = df.drop(['author','sent'], axis=1)
y = df.author

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X = count_vect.fit_transform(X.text)
X = tfidf_transformer.fit_transform(X)
X = X.toarray()
print(X.shape)
from scipy import sparse

sent = sent.reshape((sent.shape[0],1))
X = np.hstack((X, sent))
X = sparse.csr_matrix(X)
print(X.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

clf = MultinomialNB(alpha=0.1)
scores = cross_val_score(clf, X, df.author, cv=5)
print('accuracy CV:',scores)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

final = MultinomialNB(alpha=0.1)
final.fit(X, df.author)

df_t = pd.read_csv('../input/test.csv')

df_t['sent'] = df_t['text'].map(lambda x: sid.polarity_scores(x)['compound'])
df_t = clean_ColText(df_t, 'text', stem=True)
df_t['text'] = df_t.text.map(lambda x: ' '.join(x))

X_t = df_t.drop(['sent','id'], axis=1).text
X_t = count_vect.transform(X_t)
X_t = tfidf_transformer.transform(X_t)
X_t = X_t.toarray()
sent_t = np.array(df_t.sent) + 1
sent_t = sent_t.reshape((sent_t.shape[0],1))
print(X_t.shape, sent_t.shape)
X_t = np.hstack((X_t, sent_t))

X_t = sparse.csr_matrix(X_t)
print(X_t.shape)
preds = final.predict_proba(X_t)
df_t['EAP'] = preds[:,0]
df_t['HPL'] = preds[:,1]
df_t['MWS'] = preds[:,2]
df_t.head()
df_t = df_t.drop(['text', 'sent'], axis=1)
df_t.to_csv('results.csv', index=False)