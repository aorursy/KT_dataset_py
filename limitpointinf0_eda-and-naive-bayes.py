import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def clean_ColText(data, col, stem=True):
    """Takes dataframe and column name and returns a dataframe with cleaned strings in the form of a list of word tokens. 
    Stemming is an option."""
    df = data.copy()
    df[col] = df[col].map(lambda x: re.sub('\s+', ' ', x).strip())
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

def stemIt(data, col):
    df = data.copy()
    porter = PorterStemmer()
    df[col] = df[col].map(lambda x: [porter.stem(y) for y in x])
    return df

def plot_wordcloud(text, title=None, max = 1000, size=(10,5), title_size=16):
    """plots wordcloud"""
    wordcloud = WordCloud(max_words=max).generate(text)
    plt.figure(figsize=size)
    plt.title(title, size=title_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
def getNBModel(data, predictor, label, alpha=0.1):
    y = data[label]
    model = MultinomialNB(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=5)
    model = model.fit(X, y)
    return (label, model, scores) 

bp = '../input/'
print(os.listdir(bp))
df = pd.read_csv(bp + 'train.csv')
#df.head()
df = clean_ColText(df, 'comment_text', stem=False)
#df.head()
#df_eda = df.copy()
#df_eda['total'] = df_eda.iloc[:,2:].sum(axis=1)
#df_eda[df_eda['total'] >=5].head()
#txt = ' '.join(sum(list(df_eda[df_eda['total'] >=5]['comment_text']), []))
#plot_wordcloud(txt, title='Toxic Comments', size=(10,5))
targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
corr = df[targets].corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
plt.title("Toxic Comment Correlation between Types")
plt.show()
df['text'] = df['comment_text'].map(lambda x: ' '.join(x))
X = df[['text']]
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X = count_vect.fit_transform(X.text)
X = tfidf_transformer.fit_transform(X)

models_stats = []
for t in targets:
    m_s = getNBModel(df, X, t)
    models_stats.append(m_s)
    print(m_s)
df_test = pd.read_csv(bp + 'test.csv')
df_test = clean_ColText(df_test, 'comment_text', stem=False)
df_test['text'] = df_test['comment_text'].map(lambda x: ' '.join(x))

X_test = df_test[['text']]
X_test = count_vect.transform(X_test.text)
X_test = tfidf_transformer.transform(X_test)

for x in models_stats:
    df_test[x[0]] = x[1].predict_proba(X_test)[:,1]
#df_test.head()
submit = df_test.drop(['comment_text', 'text'], axis=1)
submit.to_csv('results.csv', index=False)
print('wrote to csv')