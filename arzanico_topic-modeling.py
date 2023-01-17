import pandas as pd
import numpy as np
df = pd.read_csv('../input/winemag-data-130k-v2.csv')
df.info()
#Amount of uniques varieties
len(df.variety.value_counts().index.values)
#What varieties are the most described or tested by the wine taster
count = 0
for l in df.variety.value_counts()>1000:
    if l:
        count += 1
print (count , '  Are de varieties with more than 1000 descriptions')
data = list(df.description.values)
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
no_features = 1000
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data)
tf_feature_names = tf_vectorizer.get_feature_names()
from sklearn.decomposition import NMF , LatentDirichletAllocation
no_topic = 28
#for NMF

nmf = NMF(n_components = no_topic , random_state=1 , alpha=.1 , l1_ratio=.5 , init='nndsvd').fit(tfidf)


#for LDA

lda = LatentDirichletAllocation(n_components=no_topic , max_iter=10 , learning_method='online' , learning_offset=50. , random_state=0)

lda_v = lda.fit(tf)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda_v, tf_feature_names, no_top_words)
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer, mds='tsne')
panel
