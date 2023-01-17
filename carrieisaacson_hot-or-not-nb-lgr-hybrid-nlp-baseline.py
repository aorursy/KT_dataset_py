import numpy as np

import pandas as pd

import re

import random



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import KFold

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from scipy.sparse import csr_matrix



from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

import seaborn as sns



import matplotlib

import matplotlib.colors as mc
# fictitious food reviews

docs = ['The pizza is great!',

       'This is amazingly great pizza.',

       'This is amazingly good pizza.',

       'This pizza is terribly good.',

       'These are terrible olives.',

       'These olives are terrible.',

       'These are amazingly gross olives.',

       'Olives are amazingly terrible!']

y = [1,1,1,1,0,0,0,0]
# basic text processing

# omit stop words and apply stemming

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english') + ['this', 'that', 'these'])

def process_text(text):

    # Lower case

    text = [ t.lower() for t in text.split(' ') ]

    # Drop punctuation and numerics

    text = [ re.sub(r'[^A-Za-z]', '', t) for t in text ]

    # Drop stopwords

    text = [ t for t in text if t not in stop_words ]

    # Apply porter stemming

    text = ' '.join([ stemmer.stem(t) for t in text ])

    return text

    

docs_processed = [ process_text(t) for t in docs ]



ctv = CountVectorizer()

doc_term_matrix = ctv.fit_transform(docs_processed)



x = pd.DataFrame(doc_term_matrix.todense(),

                 columns = ctv.get_feature_names())
# Some adjustements for the sake of visualization

df = x.copy()

df = df.apply(lambda x: x/3)

df['target'] = [1 if x == 0 else 0.66 for x in y]

df.index = docs



cols = ["#ffffff", "#000000", "#e74c3c", "#2ecc71"]

cm = sns.color_palette(cols)



fig, axarr = plt.subplots(1, 1, figsize=(10, 4))

sns.heatmap(df[['target'] + list(df.columns[:-1])],

            cmap=cm,

            linecolor='white',

            linewidths=2,

            cbar=False)

_ = plt.title('Bag of Words')
class NBTransformer(BaseEstimator, TransformerMixin):

    """

    Implementation of Wang, S. and Manning C.D. Baselines and Bigrams: 

    Simple, Good Sentiment and Topic Classification (2012)

    https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

    

    Adapted from: 

        Jeremy Howard Machine Learning for Coders Lesson 11

        Kaggle @jhoward https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

        Kaggle @Ren https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline

    """    

    def __init__(self, alpha=1):

        self.r = None

        self.alpha = alpha



    def fit(self, X, y):

        # Convert X matrix to 1's and 0's

        X = csr_matrix(X)

        y = csr_matrix(y).T

        y_not = csr_matrix(np.ones((y).shape[0])).T - y

        # compute smoothed log prob ratio

        # use sparse pointwise multiplication, more memory efficient

        # than slicing

        p = self.alpha + X.multiply(y).sum(0)

        q = self.alpha + X.multiply(y_not).sum(0)

        # computed using L1 norm per Wang and Manning

        # (difference is accounted for by logistic regression bias term in otherwise)

        # adjusted smoothing to be equivalent to sklearn MultinomialNB using vocab size

        # (minor deviation from Wang and Manning)

        self.r = csr_matrix(np.log(

            (p / (self.alpha * X.shape[1] + X.multiply(y).sum())) /

            (q / (self.alpha * X.shape[1] + X.multiply(y_not).sum()))

        ))

        return self



    def transform(self, X, y=None):

        return X.multiply(self.r)

    

    def fit_transform(self, X, y):

        self.fit(X, y)

        return X.multiply(self.r)
doc_term_matrix_nb = NBTransformer().fit_transform(doc_term_matrix,y)

df = pd.DataFrame(doc_term_matrix_nb.todense(),

                 columns = ctv.get_feature_names())

df['target'] = [df.min().min() if x == 0 else df.max().max() for x in y]

df.index = docs



cols = sns.diverging_palette(145, 280, s=85, l=25, n=10000)

cm = sns.color_palette(cols)



fig, axarr = plt.subplots(1, 1, figsize=(10, 4))

sns.heatmap(df[['target'] + list(df.columns[:-1])],

            cmap=cm,

            linecolor='white',

            linewidths=2)

_ = plt.title('Naive Bayes Scaled Bag of Words')
df = pd.read_csv('../input/comments.csv', header=None, names=['comment', 'rating'])



# At least a 7 sounds like a good place to be.

hotness_threshold = 7

df['hot'] = df.rating.apply(lambda x: 1 if x >= hotness_threshold else 0)



# find and remove comments that contain only a score (e.g. "8/10" or "8/10.")

df['score_only'] = df.comment.apply(lambda x: re.match(r"\d*[\.]?\d\/\d{1,2}[\.]?$",x) is not None)

df = df.loc[np.logical_not(df['score_only']),:].reset_index()

df = df.drop('score_only', axis=1)



print('{}% rated attractive (threshold score of {})'.format(int(100*df.hot.mean()), hotness_threshold))



_ = sns.distplot(df.rating,

             bins=range(0,11),

             kde=False,

             hist_kws=dict(edgecolor="w", linewidth=2))

plt.title('Distribution of Ratings')

_ = plt.axvline(7, 0,100000, color='r')
# Again applying just some basic preprocessing

def preprocessing(doc):

    # if the entirety of the comment is a numeric rating (e.g. 3/10),

    # replace with identifier token for use as feature

    if re.match(r"\d*[\.]?\d\/\d{1,2}[\.]?$", doc) is not None:

        return 'NO_COMMENT'

    # space non-alpha characters

    doc = re.sub('([^a-zA-Z]+)', r' ', doc)

    # lowercase + drop punc / numbers / special characters

    tokens = [token.lower() for token in doc.split(' ') if token.isalpha()]

    # some comments are only a rating score, lets identify those

    return ' '.join(tokens)



# Create bag of words including bigrams

cv = CountVectorizer(preprocessor=preprocessing,

                     stop_words='english',

                     analyzer='word',

                     ngram_range=(1,2),

                     max_df=0.9999,

                     min_df=0.0001)



x = cv.fit_transform(df.comment)

y = df.hot
%%time 

nb = MultinomialNB()

kf = KFold(n_splits = 4)



scores_nb = []

for it, iv in kf.split(x):

    nb.fit(x[it,:], y[it])

    probs = nb.predict_proba(x[iv,:])

    scores_nb.append(average_precision_score(y[iv], probs[:,1]))

    

fpr_nb, tpr_nb, thresholds = roc_curve(y[iv], probs[:,1])

pre_nb, rec_nb, thresholds = precision_recall_curve(y[iv], probs[:,1])
%%time 

lgr = LogisticRegression(solver='liblinear', 

                         class_weight='balanced',

                         C=0.5)



scores_lgr = []

for it, iv in kf.split(x):

    lgr.fit(x[it,:], y[it])

    probs = lgr.predict_proba(x[iv,:])

    scores_lgr.append(average_precision_score(y[iv], probs[:,1]))

    

fpr_lgr, tpr_lgr, thresholds = roc_curve(y[iv], probs[:,1])

pre_lgr, rec_lgr, thresholds = precision_recall_curve(y[iv], probs[:,1])
%%time 

# Hybrid model

pipeline = Pipeline([('NB', NBTransformer()),

                     ('Logistic Regression', LogisticRegression(solver='liblinear', 

                                                                class_weight='balanced',

                                                                C=0.5))])



scores_nblgr = []

for it, iv in kf.split(x):

    pipeline.fit(x[it,:], y[it])

    probs = pipeline.predict_proba(x[iv,:])

    scores_nblgr.append(average_precision_score(y[iv], probs[:,1]))

    

fpr_nblgr, tpr_nblgr, thresholds = roc_curve(y[iv], probs[:,1])

pre_nblgr, rec_nblgr, thresholds = precision_recall_curve(y[iv], probs[:,1])
print("Naive-Bayes: \t\t\t\t{}".format(round(np.mean(scores_nb),4)))

print("Logistic Regression: \t\t\t{}".format(round(np.mean(scores_lgr),4)))

print("Naive-Bayes + Logistic Regression: \t{}".format(round(np.mean(scores_nblgr),4)))
fig, axarr = plt.subplots(1, 2, figsize=(16, 4))

plt.subplot(121)

plt.plot(fpr_nb, tpr_nb, label='NB')

plt.plot(fpr_lgr, tpr_lgr, label='LGR')

plt.plot(fpr_nblgr, tpr_nblgr, label='NB+LGR')

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.title('ROC Curve')



plt.subplot(122)

plt.plot(rec_nb, pre_nb, label='NB')

plt.plot(rec_lgr, pre_lgr, label='LGR')

plt.plot(rec_nblgr, pre_nblgr, label='NB+LGR')

plt.xlabel('recall')

plt.ylabel('precision')

plt.title('Precision-Recall Curve')

_ = plt.legend(loc='best')