import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from string import punctuation
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
target = train.target
y_test = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

data = pd.concat([train, test])
data = data.drop(['location', 'target'], axis=1)
data.head()
class TweetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        self.new_X = X.copy()
        self.keyword_list = [i.replace('%20', '') for i in self.new_X.keyword.unique()[1:]]
        
#       Create a new feature mention, representing the usere mentioned in the tweet denotede by any word preeeded by the @ symblo
        self.new_X['mention'] = self.new_X.apply(lambda x: re.findall(r'@[a-zA-Z0-9]+', x['text']), axis=1)
        self.new_X['mention'] = self.new_X.apply(lambda x: ' '.join(x['mention']), axis=1)
#       Create a new feature to store the hash tags
        self.new_X['hashtag'] = self.new_X.apply(lambda x: re.findall(r'#\w+', x['text']), axis=1)
        self.new_X['hashtag'] = self.new_X.apply(lambda x: ' '.join(x['hashtag']), axis=1)
        # Remove #symbol from hashtags
        self.new_X['hashtag'] = self.new_X.apply(lambda x: "".join([word.lower() for word in x['hashtag'] if word not in punctuation]), axis=1)
        self.new_X['hash_to_kw'] = self.new_X.apply(lambda x: ''.join([i for i in x['hashtag'].split() if i in self.keyword_list]), axis=1)
#       Remove all hash tags from the main text data
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'#\w+','', x['text']), axis=1)
#       Since mentions have already been collected in the mention colummc, mentions should be removed from the text data
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'@[a-zA-Z0-9]+','', x['text']), axis=1)
#       remove all hyperlinks in the tweets
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', x['text']), axis=1)
#       Remove all punctuationn from the text data
        self.new_X['text'] = self.new_X.apply(lambda x: "".join([word.lower() for word in x['text'] if word not in punctuation]), axis=1)
#         new_X['tokenize_tweet'] = new_X['no_punct_tweet'].apply(lambda x: re.split(r"\W+", x), axis = 1)
#         new_X['tokenize_tweet'] = new_X.apply(lambda x: re.split(r"\W+", x['no_punct_tweet']), axis = 1)
#       Tokenize the text data
        self.new_X['text'] = self.new_X.apply(lambda x: nltk.word_tokenize(x['text']), axis=1)
#       Remove stop words from the text data
        self.new_X['text'] = self.new_X.apply(lambda x: [word for word in x['text'] if word not in stopwords], axis=1)
#       Lemmatize the text data
        self.new_X['text'] = self.new_X.apply(lambda x: [lemm.lemmatize(word) for word in x['text']], axis=1)
        self.new_X['text'] = self.new_X.apply(lambda x: ' '.join(x['text']), axis=1)
        return self.new_X
class FillMissingKeyword(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        self.new_X = X.copy()
#         self.keyword_list = list(self.new_X.keyword.unique())[1:]

#       self.new_X['keyword'] = self.new_X.apply(lambda x: re.sub(int(None), re.findall(r'#\w+', x['text'])), axis=1)
        self.new_X['Keyword'] = self.new_X['keyword'].combine_first(self.new_X['hash_to_kw']).astype('category')
#         self.new_X['keyword'] = self.new_X.apply(lambda x: ' '.join(x['keyword']), axis=1)
        return self.new_X
tweet = TweetTransformer()
fill = FillMissingKeyword()
tfidf = TfidfVectorizer(ngram_range=(1,2))
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)
transformer = Pipeline(
    steps=[
        ('tweet', tweet),
        ('cat', fill)
    ])

df = transformer.fit_transform(data)
df['Keyword'] = df.Keyword.cat.codes
df = df.drop(['mention', 'hashtag', 'hash_to_kw', 'keyword'], axis=1)
# z = qq[:7613]
text = df['text']
features = df.drop(['text'], axis=1)
tf = tfidf.fit_transform(text)
# cf = cout.fit_transform(text)
df = pd.DataFrame(tf.toarray())
features.reset_index(inplace=True)
df = pd.concat([features, df], axis=1)
X_train = df[:7613]
X_test = df[7613:]
rf = rf_model.fit(X_train, target)
y_pred = rf_model.predict(X_test)
y_test.head()
# y_test = y_test.drop(['id'], axis=1)
from sklearn.metrics import precision_recall_fscore_support as score



precision, recall, fscore, train_support = score(y_test['target'], y_pred, pos_label=1, average='binary')
print(precision, recall, fscore, train_support)
pred = pd.DataFrame(y_pred)
pred.to_csv(r'/kaggle/working/predictions.csv', header=True)
