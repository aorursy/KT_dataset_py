import pandas as pd
df = pd.DataFrame({'text': ['I like kaggle very much',

                            'I do not like kaggle',

                            'I do really love machine learning']})

df
from sklearn.feature_extraction.text import CountVectorizer





vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')

bag = vectorizer.fit_transform(df['text'])

bag.toarray()
print(vectorizer.vocabulary_)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer





vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')

transformer = TfidfTransformer()



tf = vectorizer.fit_transform(df['text'])

tfidf = transformer.fit_transform(tf)

print(tfidf.toarray())
print(vectorizer.vocabulary_)
from gensim.models import word2vec





sentences = [d.split() for d in df['text']]

model = word2vec.Word2Vec(sentences, size=10, min_count=1, window=2, seed=7)
model.wv['like']
model.wv.most_similar('like')
df['text'][0].split()
import numpy as np





wordvec = np.array([model.wv[word] for word in df['text'][0].split()])

wordvec
np.mean(wordvec, axis=0)
np.max(wordvec, axis=0)