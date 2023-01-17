from sklearn.metrics import accuracy_score

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import time

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_validate

import time

from scipy import sparse

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC

import xgboost as xgb

from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression





ps = PorterStemmer()





class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer



    def __call__(self, wo):

        return [self.wnl.lemmatize(z) for z in word_tokenize(wo)]





class Stemtfidf(TfidfVectorizer):

    def build_analyzer(self):

        analyzer = super(TfidfVectorizer, self).build_analyzer()

        return lambda doc: (ps.stem(w) for w in analyzer(doc))





def vector(input, input2):

    blorp = Stemtfidf(max_features=20000, tokenizer=nltk.word_tokenize, stop_words="english", ngram_range=(1, 4))

    tfidf = blorp.fit_transform(input['question_text'])

    print(tfidf)

    sparse.save_npz("vector.npz", tfidf)

    tfidf2 = blorp.transform(input2['question_text'])

    sparse.save_npz("vector2.npz", tfidf2)

    return tfidf, tfidf2





def plot(train_results, temp):

    train_results = np.asarray(train_results)

    print(train_results)

    print(max(train_results))

    print(np.argmax(train_results))

    line1, = plt.plot(temp, train_results, 'b', label="train")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

    plt.ylabel('f-score')

    plt.xlabel('alpha')

    plt.show()





def n_estimate(tempy, tempy2):

    tra = []

    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

    for x in range(len(n_estimators)):

        f = n_estimators[x]

        rf1 = RandomForestClassifier(n_estimators=f)

        rf1.fit(tempy, tempy2)

        tra.append(np.mean(cross_val_score(rf1, tempy, tempy2, cv=5)))

        print(x)

    plot(tra, n_estimators)





def lr(x_t, y_t, x_tes, y_tes):

    tra = []

    learn = [.01, .05, .1, .2, .3]

    for x in range(len(learn)):

        f = learn[x]

        boost = xgb.XGBClassifier(max_depth=100, n_estimators=100, learning_rate=f, colsample_bytree=.72,

                                  reg_alpha=4, objective='binary:logistic', subsample=0.75, n_jobs=-1).fit(x_t,

                                                                                                                    y_t)

        prediction = boost.predict(x_tes)

        tra.append(f1_score(y_tes, prediction, average='macro'))

        print(x)

    plot(tra, learn)





def reg(x_t, y_t, x_tes, y_tes):

    tra = []

    reg_a = [1, 2, 3, 4]

    for x in range(len(reg_a)):

        f = reg_a[x]

        boost = xgb.XGBClassifier(max_depth=100, n_estimators=100, learning_rate=0.2, colsample_bytree=.72,

                                  reg_alpha=f, objective='binary:logistic', subsample=0.75, n_jobs=-1).fit(x_t,

                                                                                                                    y_t)

        prediction = boost.predict(x_tes)

        tra.append(f1_score(y_tes, prediction, average='macro'))

        print(x)

    plot(tra, reg_a)





def alpha(x_t, y_t, x_tes, y_tes):

    tra = []

    alp = [.00001, .0001, .001, .01, .1, 1, 10, 100]

    for x in range(len(alp)):

        f = alp[x]

        bnb = BernoulliNB(alpha=f)

        bnb.fit(x_t, y_t)

        prediction = bnb.predict(x_tes)

        tra.append(f1_score(y_tes, prediction, average='macro'))

        print(x)

    plot(tra, alp)





def Cs(x_t, y_t, x_tes, y_tes):

    tra = []

    c = [.001, .01, .1, 1, 10, 100]

    for x in range(len(c)):

        f = c[x]

        svm1 = LinearSVC(C=f)

        svm1.fit(x_t, y_t)

        prediction = svm1.predict(x_tes)

        tra.append(f1_score(y_tes, prediction, average="macro"))

        print(x)

    plot(tra, c)





def svm(x_t, y_t, z_t):

    clf = LinearSVC(C=10, class_weight='balanced')

    clf.fit(x_t, y_t)

    return clf.predict(z_t)





def randforest(x_t, y_t, z_t):

    rf = RandomForestClassifier(n_estimators=100, max_depth=100, n_jobs=-1)

    rf.fit(x_t, y_t)

    return rf.predict(z_t)





def gradboost(x_t, y_t, z_t):

    xgb_model = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.2, colsample_bytree=.72,

                                  reg_alpha=3, objective='binary:logistic', subsample=0.75, n_jobs=-1).fit(x_t,

                                                                                                           y_t)

    return xgb_model.predict(z_t)





def MNB(x, y, z):

    mnb = MultinomialNB(alpha=.0001)

    mnb.fit(x, y)

    return mnb.predict(z)





def naive(x, y, z):

    nb = BernoulliNB(alpha=.0001)

    nb.fit(x, y)

    return nb.predict(z)





def linreg(x_t, y_t, z_t):

    rege = LogisticRegression()

    rege.fit(x_t, y_t)

    return rege.predict(z_t)





def mlp():

    clf = MLPClassifier()

    clf.fit(x_train, y_train)

    return clf.predict(x_test)

    # parameters = {'solver': ['lbfgs', 'adam', 'sgd'], 'max_iter': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],

    #               'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes': (10,)}

    # clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=1)

    # clf.fit(x_train, y_train)

    # print(clf.score(x_train, y_train))

    # print(clf.best_params_)=





def resample():

    return SMOTE(n_jobs=-1).fit_resample(x_train, y_train)





test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

qid = test

test = test.drop('qid', axis=1)

train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

train = train.drop('qid', axis=1)

print(train.keys())

target = train['target']

print(target.sum())



tfidf_train, tfidf_test = vector(train, test)

# tfidf_train = sparse.load_npz("vector.npz")

# tfidf_test = sparse.load_npz("vector2.npz")

print(tfidf_train.shape)

x_train, x_test, y_train, y_test = train_test_split(tfidf_train, target, test_size=0.3, random_state=27)

# x_resmamp, y_resamp = resample()

# n_estimate(x_train, y_train)

# lr(x_train, y_train, x_test, y_test)

# reg(x_train, y_train, x_test, y_test)

# alpha(x_train, y_train, x_test, y_test)

# Cs(x_train, y_train, x_test, y_test)



print("starting predict")

start = time.time()

# testy = naive(tfidf_train, target, tfidf_test)

testy = MNB(tfidf_train, target, tfidf_test)

# testy = gradboost(tfidf_train, target, tfidf_test)

# testy = linreg(tfidf_train, target, tfidf_test)

# testy = svm(tfidf_train, target, tfidf_test)

# print(testy)

# np.savetxt("numpy.txt", testy, fmt="%d")

# tes = naive(x_train, y_train, x_train)

# tes = linreg()

# tes = gradboost(x_train, y_train)

# tes = svm(x_train, y_train)

# tes = randforest(x_resmamp, y_resamp)

# tes = mlp()

# tes = naive()

print(time.time() - start)

qid = qid.drop('question_text', axis=1)

qid['prediction'] = testy

print("to_csv")

qid.to_csv('submission.csv', index=False)



# print(confusion_matrix(y_test, tes))

# print("accuracy")

# print(accuracy_score(y_test, tes))

# print("f1")

# print(f1_score(y_test, tes, average='macro'))

# print(classification_report(y_test, tes))

# print("precision")

# print(precision_score(y_test, tes))

# print("recall")

# print(recall_score(y_test, tes))

# print("cross")