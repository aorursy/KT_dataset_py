import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
df = pd.read_csv('train.csv')
df.shape
# https://www.kaggle.com/c/student-shopee-code-league-sentiment-analysis/discussion/170953
df_ex = pd.read_csv('805768_1381121_bundle_archive/shopee_reviews.csv')
df_ex = df_ex.rename(columns={'label':'rating','text':'review'})
df_ex
df = pd.concat([df,df_ex])
df = df[df['rating']!='label']
df['rating'] = df['rating'].astype(int)
df['rating'].value_counts()
class ReviewTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        word_list = X['review']\
            .apply(lambda x: x.replace('¶', ' ')\
                   .replace('&amp;', ' ')\
                   .replace('&lt;', ' ')\
                   .replace('&gt;', ' ')\
                   .replace('“', '')\
                   .replace('”', ' ')\
                   .replace('‘', ' ')\
                   .replace('’', ' ')\
                   .replace(chr(146), ' ')\
                   .replace(chr(147), ' ')\
                   .replace(chr(148), ' ')\
                   .replace(chr(160), ' ')\
                  )\
            .apply(str.lower)\
            .apply(lambda x: ' '.join(list(filter(lambda y: y[0] not in list('@&\\'), x.split()))))\
            .apply(lambda x: ' '.join(list(filter(lambda y: False if re.search('https', y, re.I) else True, x.split()))))\
            .apply(lambda x: ' '.join(list(filter(lambda y: False if re.search('http', y, re.I) else True, x.split()))))\
            .apply(lambda x: ' '.join(list(filter(lambda y: False if re.search('.com', y, re.I) else True, x.split()))))\
            .apply(lambda x: ''.join(list(
                filter(lambda y: y in list("abcdefghijklmnopqrstuvwxyz ") \
                   or any(l <= y <= u for l, u in emoji_ranges) \
                   or y in emoji_flags, x))))\
            .apply(lambda x: ' '.join(list(filter(lambda y: len(y) >= 2, x.split()))))
        return word_list
emoji_ranges = ((u'\U0001f300', u'\U0001f5ff'), (u'\U0001f600', u'\U0001f64f'), (u'\U0001f680', u'\U0001f6c5'),
                (u'\u2600', u'\u26ff'), (u'\U0001f170', u'\U0001f19a'))
print(emoji_ranges)
emoji_flags =  {u'\U0001f1ef\U0001f1f5', u'\U0001f1f0\U0001f1f7', u'\U0001f1e9\U0001f1ea',
                u'\U0001f1e8\U0001f1f3', u'\U0001f1fa\U0001f1f8', u'\U0001f1eb\U0001f1f7',
                u'\U0001f1ea\U0001f1f8', u'\U0001f1ee\U0001f1f9', u'\U0001f1f7\U0001f1fa',
                u'\U0001f1ec\U0001f1e7'}
print(emoji_flags)
any(l <= '😀' <= u for l, u in emoji_ranges)
X_train, X_test, y_train, y_test = train_test_split(df, df['rating'], test_size=0.5, random_state=0)
naive_bayes = Pipeline([
    ('transform', ReviewTransformer()),
    ('vect', TfidfVectorizer(ngram_range=(1,5), token_pattern=r'[a-z]+\'?[a-z]+|[^\s]', binary=True)),
#     ('classifier', MultinomialNB(alpha=0.072625,fit_prior=False,class_prior=[0.2,0.2,0.2,0.2,0.2]))
    ('classifier', MultinomialNB(alpha=0.072625))
])

# naive_bayes.fit(X_train, y_train)
# naive_bayes.score(X_test, y_test)
naive_bayes.named_steps['vect'].get_feature_names()
df_test = pd.read_csv('test.csv')
df_test
df_submission = df_test[['review_id']]
naive_bayes.fit(df, df['rating'])
df_submission['rating'] = naive_bayes.predict(df_test)
df_submission.to_csv('submit12.csv', index=False)
df_submission = df_test[['review_id']]
naive_bayes.fit(df, df['rating'])
df_submission['rating'] = naive_bayes.predict(df_test)
df_submission.to_csv('submit8.csv', index=False)
naive_bayes.predict(df_test.head(1))