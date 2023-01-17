import pandas as pd
data_path = "../input/Emails.csv"

data = pd.read_csv(data_path)
data.sample(5)
print(f"Number of Emails: {data.shape[0]}")
data = data[pd.notnull(data['ExtractedBodyText'])]

print(data.sample(5)['ExtractedBodyText'])
print(f"Number of Emails: {data.shape[0]}")
from nltk import RegexpTokenizer

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

texts = [tokenizer.tokenize(email.lower()) for email in data['ExtractedBodyText']]
print(texts[5070])
def delete_stopwords(tokenized_sentence: list):

    return list(filter(lambda x: x not in stop_words, tokenized_sentence))
texts = list(filter(lambda x: len(x) > 5, [delete_stopwords(text) for text in texts]))
print(f"Number of Emails: {len(texts)}")
from gensim import corpora
corpora_dict = corpora.Dictionary(texts)
print(list(corpora_dict.token2id.items())[::500])
corpora_dict[500]
corpora_dict.id2token[500]
corpus = [corpora_dict.doc2bow(text) for text in texts]
print(corpus[0])
from gensim.models import LsiModel
model_lsi = LsiModel(corpus, id2word=corpora_dict.id2token, num_topics=10)
str_topics = [topic_w for topic_number, topic_w in model_lsi.print_topics()]

str_topics_split = list(map(lambda x: x.split("+"), str_topics))

str_topics_split = [list(map(lambda x: x.split("*")[1].strip()[1:-1], elem)) for elem in str_topics_split]
for topic in str_topics_split:

    print(topic)
from gensim import matutils

from gensim.models.ldamodel import LdaModel
model_lda = LdaModel(corpus, passes=20, num_topics=10, id2word=corpora_dict.id2token)
str_topics = [topic_w for topic_number, topic_w in model_lda.print_topics()]

str_topics_split = list(map(lambda x: x.split("+"), str_topics))

str_topics_split = [list(map(lambda x: x.split("*")[1].strip()[1:-1], elem)) for elem in str_topics_split]



for topic in str_topics_split:

    print(topic)
import pyLDAvis
import pyLDAvis.gensim



data_lda = pyLDAvis.gensim.prepare(model_lda, corpus, corpora_dict)
pyLDAvis.enable_notebook()

pyLDAvis.display(data_lda)
from gensim.matutils import corpus2dense

import numpy as np

np.random.seed(42)
class PlsaModel:

    def __init__(self, corpus=None, id2word=None, num_topics=10, passes=30):

        self.passes = passes

        

        self.num_topics = num_topics

        self.num_documents = len(corpus)

        self.num_words = len(id2word)



        self.id2word = id2word

        

        self.n_wd = corpus2dense(corpus, num_terms=self.num_words)  # [word][document]

        self.n_d = np.sum(self.n_wd, axis=0)

        self.n = np.sum(self.n_d)



        self.phi = np.random.random_sample(size=(self.num_words, self.num_topics))

        self.phi /= np.sum(self.phi, axis=0)

        self.theta_t = np.random.random_sample(size=(self.num_documents, self.num_topics))

        self.theta_t /= np.sum(self.theta_t, axis=1)[:, None]

        

        for i in range(self.passes):

            self._fit()



    def _fit(self):

        # n_zd = # YOUR CODE HERE

        # n_wz = # YOUR CODE HERE

        # n_z = # YOUR CODE HERE

        for d in range(self.num_documents):

            # YOUR CODE HERE

            pass



        # YOUR CODE HERE

        

    def print_topics(self, top_n=10):

        res = []

        for t in range(self.num_topics):

            top_inds = self.phi[:, t].argsort()[-top_n:][::-1]

            top_words = [self.id2word[x] for x in top_inds]

            res.append(top_words)

        return res
model_plsa = PlsaModel(corpus, passes=10, num_topics=10, id2word=corpora_dict.id2token)
for topic in model_plsa.print_topics():

    print(topic)