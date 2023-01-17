import pandas as pd

import numpy as np



import json



import spacy

from spacy import displacy

from collections import Counter

import en_core_web_lg



nlp = en_core_web_lg.load()
import re

import string



from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.calibration import CalibratedClassifierCV

from imblearn.under_sampling import InstanceHardnessThreshold

from sklearn.svm import LinearSVC

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.svm import LinearSVC



from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline
import seaborn as sns

sns.set()

from IPython.core.pylabtools import figsize

figsize(20, 20)
# Just disable some annoying warning

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn
df = pd.read_csv('../input/newsgroup20bbcnews/news_group20.csv')
df.count()
df.category.value_counts().plot.barh()
import re

import string



from sklearn.base import TransformerMixin



class TextPreprocessor(TransformerMixin):

    def __init__(self, text_attribute):

        self.text_attribute = text_attribute

        

    def transform(self, X, *_):

        X_copy = X.copy()

        X_copy[self.text_attribute] = X_copy[self.text_attribute].apply(self._preprocess_text)

        return X_copy

    

    def _preprocess_text(self, text):

        return self._lemmatize(self._leave_letters_only(self._clean(text)))

    

    def _clean(self, text):

        bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'

        text_without_symbols = text.translate(str.maketrans('', '', bad_symbols))



        text_without_bad_words = ''

        for line in text_without_symbols.split('\n'):

            if not line.lower().startswith('from:') and not line.lower().endswith('writes:'):

                text_without_bad_words += line + '\n'



        clean_text = text_without_bad_words

        email_regex = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'

        regexes_to_remove = [email_regex, r'Subject:', r'Re:']

        for r in regexes_to_remove:

            clean_text = re.sub(r, '', clean_text)



        return clean_text

    

    def _leave_letters_only(self, text):

        text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))

        return ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))

    

    def _lemmatize(self, text):

        doc = nlp(text)

        words = [x.lemma_ for x in [y for y in doc if not y.is_stop and y.pos_ != 'PUNCT' 

                                    and y.pos_ != 'PART' and y.pos_ != 'X']]

        return ' '.join(words)

    

    def fit(self, *_):

        return self
text_preprocessor = TextPreprocessor(text_attribute='text')

df_preprocessed = text_preprocessor.transform(df)
from sklearn.model_selection import train_test_split



train, test = train_test_split(df_preprocessed, test_size=0.3)
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vectorizer = TfidfVectorizer(analyzer = "word", max_features=10000)



X_tfidf_train = tfidf_vectorizer.fit_transform(train['text'])

X_tfidf_test = tfidf_vectorizer.transform(test['text'])
y = train['category']

y_test = test['category']
from sklearn.calibration import CalibratedClassifierCV

from imblearn.under_sampling import InstanceHardnessThreshold

from sklearn.svm import LinearSVC



iht = InstanceHardnessThreshold(random_state=0, n_jobs=11,

                                 estimator=CalibratedClassifierCV(

                                     LinearSVC(C=100, penalty='l1', max_iter=500, dual=False)

                                 ))

X_resampled, y_resampled = iht.fit_resample(X_tfidf_train, y)

print(sorted(Counter(y_resampled).items()))
print("Dataset shape: ", X_resampled.shape)
X, y = X_resampled, y_resampled

X_test, y_test = X_tfidf_test, y_test
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_norm = scaler.fit_transform(X.toarray())

X_test_norm = scaler.transform(X_test.toarray())
from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=100, penalty='l1', max_iter=500, dual=False)

lsvc.fit(X_norm, y)

fs = SelectFromModel(lsvc, prefit=True)

X_selected = fs.transform(X_norm)

X_test_selected = fs.transform(X_test_norm)
from IPython.display import Markdown, display



def show_top10_features(classifier, feature_names, categories):

    for i, category in enumerate(categories):

        top10 = np.argsort(classifier.coef_[i])[-10:]

        display(Markdown("**%s**: %s" % (category, ", ".join(feature_names[top10]))))
feature_names = np.array(tfidf_vectorizer.get_feature_names())

show_top10_features(lsvc, feature_names, lsvc.classes_)
print("New dataset shape: ", X_selected.shape)

print("Features reducted: ", X_norm.shape[1] - X_selected.shape[1])
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.svm import LinearSVC



from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
# this snippet was taken from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



def print_confusion_matrix(confusion_matrix, 

                           class_names, 

                           figsize = (15,15), 

                           fontsize=12,

                           ylabel='True label',

                           xlabel='Predicted label'):

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    

    Arguments

    ---------

    confusion_matrix: numpy.ndarray

        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 

        Similarly constructed ndarrays can also be used.

    class_names: list

        An ordered list of class names, in the order they index the given confusion matrix.

    figsize: tuple

        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,

        the second determining the vertical size. Defaults to (10,7).

    fontsize: int

        Font size for axes labels. Defaults to 14.

        

    Returns

    -------

    matplotlib.figure.Figure

        The resulting confusion matrix figure

    """

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel(ylabel)

    plt.xlabel(xlabel)
def evaluate_model(model, X, y, X_test, y_test, target_names=None):

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    scores_test = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

    

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    print("Accuracy test: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std()))

    

    print("Test classification report: ")

    if target_names is None:

        target_names = model.classes_

    print(classification_report(y_test, model.predict(X_test), target_names=target_names))

    print("Test confusion matrix: ")

    print_confusion_matrix(confusion_matrix(y_test, model.predict(X_test)), class_names=target_names)
mb = MultinomialNB()

mb.fit(X_selected, y_resampled)

evaluate_model(mb, X_selected, y, X_test_selected, y_test)
cb = ComplementNB()

cb.fit(X_selected, y_resampled)

evaluate_model(cb, X_selected, y, X_test_selected, y_test)
lr = LogisticRegression(C=10000, penalty='l1', multi_class='ovr')

lr.fit(X_selected, y)

evaluate_model(lr, X_selected, y, X_test_selected, y_test)
lsvc = LinearSVC(C=1000, penalty='l1', max_iter=500, dual=False)

lsvc.fit(X_selected, y)

evaluate_model(lsvc, X_selected, y, X_test_selected, y_test)
sgd = SGDClassifier(alpha=.0001, max_iter=50, loss='log',

                                       penalty="elasticnet", n_jobs=-1)

sgd.fit(X_selected, y)

evaluate_model(sgd, X_selected, y, X_test_selected, y_test)
vclf_sgd = VotingClassifier(estimators=[

         ('lr', LogisticRegression(C=10000, penalty='l1', multi_class='ovr')),

        ('mb', MultinomialNB()),

        ('sgd', SGDClassifier(alpha=.0001, max_iter=50, loss='log',

                                       penalty="elasticnet"))

], voting='soft', n_jobs=-1)

vclf_sgd.fit(X_selected, y)

evaluate_model(vclf_sgd, X_selected, y, X_test_selected, y_test)
clfs = (('ComplementNB', cb), 

        ('MultinomialNB', mb),

        ('LinearSVC', lsvc),

        ('LogisticRegression', lr),

        ('SGDClassifier', sgd),

        ('VotingClassifier', vclf_sgd))



for _x, _y in ((0,0), (4,4), (18,18), (19,19)):

    mtx = np.zeros((len(clfs), (len(clfs))), dtype=int)



    for i, (label1, clf1) in enumerate(clfs):

        for j, (label2, clf2) in enumerate(clfs):

            mtx[i][j] = (confusion_matrix(y_test, clf1.predict(X_test_selected))

                            -confusion_matrix(y_test, clf2.predict(X_test_selected)))[_x][_y]

    display(Markdown(f"Differect of correctly classified: **{clf2.classes_[_x]}**"))

    print_confusion_matrix(mtx, class_names=[l for l, _ in clfs], xlabel="", ylabel="", figsize=(5,5))

    plt.show()



mtx = np.zeros((len(clfs), (len(clfs))), dtype=int)

for i, (label1, clf1) in enumerate(clfs):

    for j, (label2, clf2) in enumerate(clfs):

        mtx[i][j] = (confusion_matrix(y_test, clf1.predict(X_test_selected))

                        -confusion_matrix(y_test, clf2.predict(X_test_selected))).diagonal().sum()

display(Markdown(f"**Overall** correctly classified"))

print_confusion_matrix(mtx, class_names=[l for l, _ in clfs], xlabel="", ylabel="", figsize=(5,5))
import re

import string



from sklearn.base import TransformerMixin, BaseEstimator



class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, text_attribute):

        self.text_attribute = text_attribute

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, *_):

        X_copy = X.copy()

        return X_copy[self.text_attribute].apply(self._preprocess_text)

    

    def _preprocess_text(self, text):

        return self._lemmatize(self._leave_letters_only(self._clean(text)))

    

    def _clean(self, text):

        bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'

        text_without_symbols = text.translate(str.maketrans('', '', bad_symbols))



        text_without_bad_words = ''

        for line in text_without_symbols.split('\n'):

            if not line.lower().startswith('from:') and not line.lower().endswith('writes:'):

                text_without_bad_words += line + '\n'



        clean_text = text_without_bad_words

        email_regex = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'

        regexes_to_remove = [email_regex, r'Subject:', r'Re:']

        for r in regexes_to_remove:

            clean_text = re.sub(r, '', clean_text)



        return clean_text

    

    def _leave_letters_only(self, text):

        text_without_punctuation = text.translate(str.maketrans('', '', string.punctuation))

        return ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))

    

    def _lemmatize(self, text):

        doc = nlp(text)

        words = [x.lemma_ for x in [y for y in doc if not y.is_stop and y.pos_ != 'PUNCT' 

                                    and y.pos_ != 'PART' and y.pos_ != 'X']]

        return ' '.join(words)
class DenseTransformer(TransformerMixin):



    def fit(self, X, y=None, **fit_params):

        return self



    def transform(self, X, y=None, **fit_params):

        return X.todense()
from imblearn.pipeline import Pipeline



text_classification_pipeline = Pipeline([

    ('text_preprocessor', TextPreprocessor(text_attribute='text')),

    ('vectorizer', TfidfVectorizer(analyzer = "word", max_features=10000)),

    ('balancer', InstanceHardnessThreshold(n_jobs=-1,

                                 estimator=CalibratedClassifierCV(

                                     LinearSVC(C=100, penalty='l1', max_iter=500, dual=False)

                                 ))),

    ('todense_converter', DenseTransformer()),

    ('scaler', MinMaxScaler()),

    ('feature_selector', SelectFromModel(LinearSVC(C=100, penalty='l1', max_iter=500, dual=False), prefit=False)),

    ('classifier', VotingClassifier(estimators=[

                         ('lr', LogisticRegression(C=10000, penalty='l1', multi_class='ovr')),

                         ('mb', MultinomialNB()),

                         ('sgd', SGDClassifier(alpha=.0001, max_iter=50, loss='log', penalty="elasticnet"))

                    ], voting='soft', n_jobs=-1))

])
parent_cats = (

    ('comp', ('comp.graphics',

            'comp.os.ms-windows.misc',

            'comp.sys.ibm.pc.hardware',

            'comp.sys.mac.hardware',

            'comp.windows.x')),

    ('foresale', ('misc.forsale',)),

    ('rec', ('rec.autos',

            'rec.motorcycles',

            'rec.sport.baseball',

            'rec.sport.hockey')),

    ('talk', ('talk.politics.misc',

            'talk.politics.guns',

            'talk.politics.mideast')),

    ('sci', ('sci.crypt',

            'sci.electronics',

            'sci.med',

            'sci.space')),

    ('religion', ('talk.religion.misc',

                'alt.atheism',

                'soc.religion.christian'))

)
df_parent = df



for new_name, lst in parent_cats:

    df_parent.loc[df_parent['category'].isin(lst), 'category'] = new_name
train, test = train_test_split(df_parent, test_size=0.3)



X_p = train.drop(columns=['category'])

y_p = train['category']



X_p_test = test.drop(columns=['category'])

y_p_test = test['category']
%%time

pipeline = text_classification_pipeline

pipeline.fit(X_p, y_p)
%%time

y_pred = pipeline.predict(X_p)
%%time

y_test_pred = pipeline.predict(X_p_test)
print(classification_report(y_p, y_pred, target_names=pipeline.classes_))

print_confusion_matrix(confusion_matrix(y_p, y_pred), class_names=pipeline.classes_, figsize=(5,5), fontsize=12)
print(classification_report(y_p_test, y_test_pred, target_names=pipeline.classes_))

print_confusion_matrix(confusion_matrix(y_p_test, y_test_pred), class_names=pipeline.classes_, figsize=(5,5), fontsize=12)
df_bbc = pd.read_csv('../input/newsgroup20bbcnews/bbc-text.csv')
df_bbc.category.value_counts().plot.barh()
train, test = train_test_split(df_bbc, test_size=0.3)



X_bbc = train.drop(columns=['category'])

y_bbc = train['category']



X_bbc_test = test.drop(columns=['category'])

y_bbc_test = test['category']
%%time

pipeline = text_classification_pipeline

pipeline.fit(X_bbc, y_bbc)
%%time

y_pred = pipeline.predict(X_bbc)
%%time

y_test_pred = pipeline.predict(X_bbc_test)
print(classification_report(y_bbc, y_pred, target_names=pipeline.classes_))

print_confusion_matrix(confusion_matrix(y_bbc, y_pred), class_names=pipeline.classes_, figsize=(5,5), fontsize=12)
print(classification_report(y_bbc_test, y_test_pred, target_names=pipeline.classes_))

print_confusion_matrix(confusion_matrix(y_bbc_test, y_test_pred), class_names=pipeline.classes_, figsize=(5,5), fontsize=12)