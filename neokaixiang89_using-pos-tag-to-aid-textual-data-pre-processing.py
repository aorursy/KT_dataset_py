import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, FreqDist
df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
df.head()
df = df.replace(np.nan, "")
df.head()
df.shape
df.dtypes
df.describe()
tokens = word_tokenize(df['Review Text'].to_string())
len(tokens)
len(set(tokens))
fd = nltk.FreqDist(tokens)
fd.plot(30)
# WNlemma = nltk.WordNetLemmatizer()

# def pre_process(text):
#     text = text.lower()
#     tokens = nltk.word_tokenize(text)
#     tokens = [t for t in tokens if len(t) > 2]
#     tokens = [WNlemma.lemmatize(t) for t in tokens]
#     text_after_process = " ".join(tokens)
#     return text_after_process
WNlemma = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk import pos_tag

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def pre_process_with_pos_tag(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [WNlemma.lemmatize(t, get_wordnet_pos(pos_tag(word_tokenize(t))[0][1])) for t in tokens]
    text_after_process = " ".join(tokens)
    return text_after_process
# review_text_processed = df['Review Text'].apply(pre_process)
review_text_processed = df['Review Text'].apply(pre_process_with_pos_tag)
review_text_processed
review_text_processed_df = pd.DataFrame(
    {
        'review_text_processed':review_text_processed,
        'recommended':df['Recommended IND']
    },
    columns = ['review_text_processed','recommended']
)
review_text_processed_df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(review_text_processed_df.review_text_processed,
                                                    review_text_processed_df.recommended,
                                                    test_size = 0.2,
                                                    random_state = 5205
                                                   )
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape
dtm = pd.DataFrame(X_train_counts.toarray().transpose(), index=count_vect.get_feature_names())
dtm = dtm.transpose()
dtm.head()
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

nb_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())
                  ])
nb_clf.fit(X_train, y_train)
nb_predicted = nb_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, nb_predicted))
print(np.mean(nb_predicted==y_test))
print(metrics.classification_report(y_test, nb_predicted))
from sklearn import tree
dt_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', tree.DecisionTreeClassifier())
                  ])
dt_clf.fit(X_train, y_train)
dt_predicted = dt_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, dt_predicted))
print(np.mean(dt_predicted==y_test))
print(metrics.classification_report(y_test, dt_predicted))
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

svm_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svm.LinearSVC(C=1.0))
                   ])
svm_clf.fit(X_train, y_train)
svm_predicted = svm_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, svm_predicted))
print(np.mean(svm_predicted==y_test))
print(metrics.classification_report(y_test, svm_predicted))
from sklearn.linear_model import LogisticRegression
lr_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression())
                  ])
lr_clf.fit(X_train, y_train)
lr_predicted = lr_clf.predict(X_test)

print(metrics.confusion_matrix(y_test, lr_predicted))
print(np.mean(lr_predicted==y_test))
print(metrics.classification_report(y_test, lr_predicted))
lr_clf = LogisticRegression()
# lr_clf.fit(dtm, y_train)
lr_clf.fit(X_train_tf, y_train)

lr_clf_coef = (
    pd.DataFrame(lr_clf.coef_[0], index=dtm.columns)
    .rename(columns={0:'Coefficient'})
    .sort_values(by='Coefficient', ascending=False)
)

lr_clf_coef.head()