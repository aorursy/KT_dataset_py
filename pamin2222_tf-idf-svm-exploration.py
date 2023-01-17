# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import unicode_literals

import os, sys
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()
%matplotlib inline

#pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 999
train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")
train.head()
phrase_id_col_name = "PhraseId"
phrase_col_name = "Phrase"
sentiment_col_name = "Sentiment"
train.describe()
test.describe()
train_text_phrases = train[phrase_col_name].tolist()
train_sentiment_labels = train[sentiment_col_name].tolist()

test_text_phrases = test[phrase_col_name].tolist()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=5, max_features=16000, strip_accents='unicode', lowercase=True,
    analyzer='word', ngram_range=(1, 3), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, tokenizer=TweetTokenizer().tokenize, stop_words='english'
)
tfidf_vectorizer.fit(train_text_phrases)
plt.figure(figsize=(12,8))
sns.countplot(x=sentiment_col_name, data=train)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Class Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Classes", fontsize=15)
plt.show()
train["text_num_words"] = train[phrase_col_name].apply(lambda x: len(TweetTokenizer().tokenize(x)))
train["text_num_chars"] = train[phrase_col_name].apply(lambda x: len(str(x)) )
numWords = []
for index, row in train.iterrows():
        counter = 0
        num_words = row["text_num_words"]
        counter = counter + num_words
        numWords.append(counter)
numSentences = len(numWords)
print('The total number of sentences is', numSentences)
print('The average number of words in the training sentences is', sum(numWords)/numSentences)
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 60, 0, 28000])
plt.show()
plt.figure(figsize=(12, 8))
sns.distplot(train["text_num_words"].values, bins=50, kde=False, color='red')
plt.xlabel('Number of words in text', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Frequency of number of words", fontsize=15)
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(x=sentiment_col_name, y='text_num_words', data=train)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Text - Number of words', fontsize=12)
plt.show()
def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
class_Xtr = tfidf_vectorizer.transform(train[phrase_col_name])
class_y = train[sentiment_col_name]
class_features = tfidf_vectorizer.get_feature_names()
class_top_dfs = top_feats_by_class(class_Xtr, class_y, class_features)
plot_tfidf_classfeats_h(class_top_dfs, 7)
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score, classification_report
import scikitplot.plotters as skplt
%matplotlib inline
X_train_tfidfmatrix = tfidf_vectorizer.transform(train_text_phrases)
X_test_tfidfmatrix = tfidf_vectorizer.transform(test_text_phrases)
def evaluate_features(X, y, clf=None):
    """General helper function for evaluating effectiveness of passed features in ML model
    
    Prints out Log loss, accuracy with 5-fold stratified cross-validation
    
    Args:
        X (array-like): Features array. Shape (n_samples, n_features)
        
        y (array-like): Labels array. Shape (n_samples,)
        
        clf: Classifier to use. If None, default Log reg is use.
    """
    if clf is None:
        clf = LogisticRegression()
    
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=5, random_state=8), 
                              n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    print('F1-micro score: {}'.format(f1_score(y, preds, average="micro")))
    skplt.plot_confusion_matrix(y, preds)
    print(classification_report(y, preds))
    return preds
# log_reg_preds = evaluate_features(X_train_tfidfmatrix, train_sentiment_labels, clf=LogisticRegression())
# svm_preds = evaluate_features(X_train_tfidfmatrix, train_sentiment_labels, clf=svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
#   max_iter=-1, probability=True, random_state=None, shrinking=True,
#   tol=0.001, verbose=False))
# from sklearn.model_selection import GridSearchCV
# parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 2, 3, 5, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(X=X_train_tfidfmatrix, y=train_sentiment_labels)

# sorted(clf.cv_results_.keys())
# clf.cv_results_
# clf.best_estimator_
svm_clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svm_clf.fit(X=X_train_tfidfmatrix, y=train_sentiment_labels)
svm_test_pred = svm_clf.predict(X_test_tfidfmatrix)
sub['Sentiment'] = svm_test_pred
sub.to_csv("submission_tfidf_svm.csv", index=False)
