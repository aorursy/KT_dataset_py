from sklearn.datasets import fetch_20newsgroups



categories = ['alt.atheism', 'soc.religion.christian', 

              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(

    subset='train',

    categories=categories,

    shuffle=True,

    random_state=42,

    remove=('headers', 'footers'),

)

twenty_test = fetch_20newsgroups(

    subset='test',

    categories=categories,

    shuffle=True,

    random_state=42,

    remove=('headers', 'footers'),

)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline, make_pipeline



vec = TfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1, 2))

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)

lsa = make_pipeline(vec, svd)



clf = SVC(C=150, gamma=2e-2, probability=True)

pipe = make_pipeline(lsa, clf)

pipe.fit(twenty_train.data, twenty_train.target)

pipe.score(twenty_test.data, twenty_test.target)
def print_prediction(doc):

    y_pred = pipe.predict_proba([doc])[0]

    for target, prob in zip(twenty_train.target_names, y_pred):

        print("{:.3f} {}".format(prob, target))    



doc = twenty_test.data[0]

print_prediction(doc)
import eli5

from eli5.lime import TextExplainer



te = TextExplainer(random_state=42)

te.fit(doc, pipe.predict_proba)

te.show_prediction(target_names=twenty_train.target_names)
import re

doc2 = re.sub(r'(recall|kidney|stones|medication|pain|tech)', '', doc, flags=re.I)

print_prediction(doc2)
print(te.samples_[0])
len(te.samples_)
te.metrics_
import numpy as np



def predict_proba_len(docs):

    # nasty predict_proba - the result is based on document length,

    # and also on a presence of "medication"

    proba = [

        [0, 0, 1.0, 0] if len(doc) % 2 or 'medication' in doc else [1.0, 0, 0, 0] 

        for doc in docs

    ]

    return np.array(proba)    



te3 = TextExplainer().fit(doc, predict_proba_len)

te3.show_prediction(target_names=twenty_train.target_names)
te3.metrics_
from sklearn.pipeline import make_union

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import TransformerMixin



class DocLength(TransformerMixin):

    def fit(self, X, y=None):  # some boilerplate

        return self

    

    def transform(self, X):

        return [

            # note that we needed both positive and negative 

            # feature - otherwise for linear model there won't 

            # be a feature to show in a half of the cases

            [len(doc) % 2, not len(doc) % 2] 

            for doc in X

        ]

    

    def get_feature_names(self):

        return ['is_odd', 'is_even']



vec = make_union(DocLength(), CountVectorizer(ngram_range=(1,2)))

te4 = TextExplainer(vec=vec).fit(doc[:-1], predict_proba_len)



print(te4.metrics_)

te4.explain_prediction(target_names=twenty_train.target_names)
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.linear_model import SGDClassifier



vec_char = HashingVectorizer(analyzer='char_wb', ngram_range=(4,5))

clf_char = SGDClassifier(loss='log')



pipe_char = make_pipeline(vec_char, clf_char)

pipe_char.fit(twenty_train.data, twenty_train.target)

pipe_char.score(twenty_test.data, twenty_test.target)
eli5.show_prediction(clf_char, doc, vec=vec_char,

                    targets=['sci.med'], target_names=twenty_train.target_names)
te = TextExplainer(random_state=42).fit(doc, pipe_char.predict_proba)

print(te.metrics_)

te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)
te = TextExplainer(char_based=True, n_samples=50000, random_state=42)

te.fit(doc, pipe_char.predict_proba)

print(te.metrics_)

te.show_prediction(targets=['sci.med'], target_names=twenty_train.target_names)