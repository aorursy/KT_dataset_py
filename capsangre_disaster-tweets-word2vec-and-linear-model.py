import pandas as pd
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
submiss = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
train['target'].unique()
from sklearn import preprocessing, feature_extraction, linear_model, model_selection
import spacy
import numpy as np
import re
from spacy.tokenizer import Tokenizer
prefix_re = re.compile('''^\$[a-zA-Z0-9]''')

def custom_tokenizer(nlp):
    """Word extraction tokenizer."""
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)
def process_text(array):
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp)
    docs = []
    for sentence in array:
        tokens = nlp.tokenizer(sentence)
        docs.append(list(map(lambda x: x.lemma_, tokens)))
    return docs
token_text = process_text(train['text'])
token_text
class MeanEmbeddingVectorizer(object):
    """Vector to mean."""

    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.vector_size

    def fit(self): 
        return self

    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.vocab:
                mean.append(self.word_model.get_vector(word))

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])
path = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
import gensim
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2v_model)
mean_embedded = mean_embedding_vectorizer.transform(token_text)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(mean_embedded, train['target'], test_size=0.25, random_state=7)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
model = linear_model.LogisticRegression()
model.fit(train_x, train_y)
metrics.accuracy_score(model.predict(test_x), test_y)