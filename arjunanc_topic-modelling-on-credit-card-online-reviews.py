# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
metadata=pd.read_excel('../input/All_Reviews.xlsx')
metadata.head(3)
metadata['Reviews'].head(5)
metadata['Reviews'] = metadata['Reviews'].fillna('')
metadata['Reviews'].head(10)
No_of_seleced_doc=7500
df1=metadata['Reviews'].iloc[0:No_of_seleced_doc]
df1.head(20)
df1.shape
print("There are {} review comments from {} different categories, such as {}... \n".format(df1.shape[0],len(metadata.Category.unique()),", ".join(metadata.Category.unique()[0:5])))
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(str(review) for review in metadata['Reviews'])
print ("There are {} words in the combination of all reviews.".format(len(text)))
stopwords = set(STOPWORDS)
stopwords.update(["Nan","Negative","etc","got","credit card","card","will"])
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(text)
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
import os
import gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
data_words = list(sent_to_words(df1))
print(data_words[0])
print(data_words[:2])
import re, nltk, spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_words[:2])
print(data_lemmatized[:2])
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf_vectorizer=TfidfVectorizer(analyzer='word',       
                             min_df=10,                        # minimum read occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{2,}', )
tfidf=tfidf_vectorizer.fit_transform(data_lemmatized)
tfidf.shape
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#tfidf_feature_names
tf_vectorizer=CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum read occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                            )
tf=tf_vectorizer.fit_transform(data_lemmatized)
tf.shape
tf_feature_names = tf_vectorizer.get_feature_names()
#tf_feature_names
tf
print("Sparsicity in tf matrix: ", ((tf > 0).sum()/tf.size)*100, "%")
data_dense = tf.todense()
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
from sklearn.decomposition import NMF, LatentDirichletAllocation
no_topics = 5
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

lda_model = LatentDirichletAllocation(n_components=no_topics,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='batch',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(tfidf)
print(lda_model) 
print("Log Likelihood: ", lda_model.score(tf))
print("Perplexity: ", lda_model.perplexity(tf))
search_params = {'n_components': [3, 4, 6, 7], 'learning_decay': [.4,.45, .55]}

lda = LatentDirichletAllocation()
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(lda, param_grid=search_params)

model.fit(tf)
best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(tf))
#gscore=model.fit(tf).cv_results_
print(model.scorer_)
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer,mds='tsne') #no other mds function like tsne used.
panel
lda_output = lda_model.transform(tfidf)
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data_lemmatized))]
import numpy as np
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic

def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)
def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)

df_document_topics
df_document_topic.info()
panel = pyLDAvis.sklearn.prepare(lda_model, tf, tf_vectorizer,mds='tsne') #no other mds function like tsne used.

panel
df_topic_keywords = pd.DataFrame(lda_model.components_/lda_model.components_.sum(axis=1)[:,np.newaxis])

df_topic_keywords.columns = tfidf_vectorizer.get_feature_names()

df_topic_keywords.index = topicnames
df_topic_keywords.head(15)
def show_lda_topics(lda_model=lda_model, n_words=20):
    keywords = np.array(df_topic_keywords.columns)
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_lda_topics(lda_model=lda_model, n_words=15)
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

df_topic_keywords
no_top_words = 8
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
display_topics(nmf, tfidf_feature_names, no_top_words)