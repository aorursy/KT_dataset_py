import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import re



papers = pd.read_csv('../input/papers.csv')

papers.head()
papers = papers.drop(["id", "event_type", "pdf_name"], axis = 1)



papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))

papers['title_processed'] = papers['title_processed'].map(str.lower)
data = [title.split() for title in papers['title_processed']]



from nltk.corpus import stopwords

stop_words = stopwords.words('english')



import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess



def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



data = remove_stopwords(data)

id2word = corpora.Dictionary(data)

corpus = [id2word.doc2bow(text) for text in data]
from gensim.models import CoherenceModel



def compute_coherence_values(dictionary, data, corpus, limit, start, step):

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,

                                           id2word = dictionary,

                                           num_topics = num_topics,

                                           random_state = 10)

        model_list.append(model)

        coherencemodel = CoherenceModel(model = model,

                                        texts = data, 

                                        dictionary = dictionary,

                                        coherence = 'c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
start = 10; limit = 30; step = 2



model_list, coherence_values = compute_coherence_values(dictionary = id2word, 

                                                        data = data, corpus = corpus, 

                                                        start = start, limit = limit,

                                                        step = step)
x = range(start, limit, step)

plt.figure(figsize = (13, 4))

plt.plot(x, coherence_values, color = 'indigo')

plt.xlabel("Num Topics", fontsize = 13)

plt.ylabel("Coherence score", fontsize = 13)

plt.xticks(x)

plt.show()
for m, cv in zip(x, coherence_values):

    print("Num Topics =", m, " has Coherence Value of", round(cv, 3))
optimal_model = model_list[5]

model_topics = optimal_model.show_topics(20, formatted = False)

for i in range(20):

    print((model_topics[i][0] + 1), (list(model_topics[i][1][j][0] for j in range(3))))
import warnings

warnings.simplefilter("ignore", FutureWarning)



from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words = 'english')

count_data = count_vectorizer.fit_transform(papers['title_processed'])



from sklearn.decomposition import LatentDirichletAllocation as LDA

lda = LDA(n_components = 20, random_state = 10)

model = lda.fit(count_data)



import pyLDAvis

import pyLDAvis.sklearn

pyLDAvis.enable_notebook()

vis = pyLDAvis.sklearn.prepare(model, count_data, count_vectorizer, mds='tsne')

saved = pyLDAvis.save_html(vis, fileobj = "vis.html")