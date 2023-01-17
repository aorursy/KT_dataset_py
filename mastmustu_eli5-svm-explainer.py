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
print(twenty_train.DESCR)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline, make_pipeline



vec = TfidfVectorizer(min_df=3, stop_words='english',

                      ngram_range=(1, 2) , max_features = 2000)

#svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)

lsa = make_pipeline(vec)



clf = SVC(C=150, gamma=2e-2, probability=True)

pipe = make_pipeline(lsa, clf)

pipe.fit(twenty_train.data, twenty_train.target)

pipe.score(twenty_test.data, twenty_test.target)
def print_prediction(doc):

    y_pred = pipe.predict_proba([doc])[0]

    for target, prob in zip(twenty_train.target_names, y_pred):

        print("{:.3f} {}".format(prob, target))



doc = twenty_test.data[20]

print_prediction(doc)
print(doc)
import eli5

from eli5.lime import TextExplainer



te = TextExplainer(random_state=42)

te.fit(doc, pipe.predict_proba)

te.show_prediction(target_names=twenty_train.target_names)