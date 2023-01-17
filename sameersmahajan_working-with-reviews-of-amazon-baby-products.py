import pandas as pd
products = pd.read_csv('../input/amazon_baby.csv')
len(products)
products.head()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

word_count = vectorizer.fit_transform(products['review'].values.astype('U'))
print (word_count.shape)
products['sentiment'] = products['rating'] >= 4
products.head()
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(word_count, products['sentiment'])
clf.coef_.shape
clf.coef_
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

text_clf = Pipeline([('vect', CountVectorizer()),

                     ('tfidf', TfidfTransformer()),

                     ('clf', MultinomialNB()), ])
text_clf.fit(products.review.values.astype('U'), products.sentiment)
products['prediction'] = text_clf.predict_proba(products.review.values.astype('U'))[:, [True, False]]
giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
giraffe_reviews = giraffe_reviews.sort_values('prediction')
giraffe_reviews.head()
pd.options.display.max_colwidth = 1000

giraffe_reviews.iloc[0]['review']
giraffe_reviews.iloc[1]['review']
giraffe_reviews.iloc[-1]['review']
giraffe_reviews.iloc[-2]['review']