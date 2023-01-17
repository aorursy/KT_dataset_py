!pip install pyvi
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve, auc
skf = StratifiedKFold(n_splits=5)
df = pd.read_csv("../input/isods-dataset/training_data_sentiment.csv")

df.tail()
df.shape
sns.countplot(df.is_unsatisfied)
print(f'"Y" chiếm {round(df[df["is_unsatisfied"] == "Y"].shape[0] / df.shape[0] * 100, 2)}% tổng dữ liệu!')
from pyvi.ViTokenizer import tokenize

import string

import re





def remove_stop_words(text):

    res = []

    with open("../input/isods-dataset/stop_words.txt") as f:

        stop_words = f.read().splitlines()

    for word in text.split():

        if word not in stop_words:

            res.append(word)



    return " ".join(res)

    



def normalize_text(text):

    res = []

    has_char_re = r".*[A-z].*"

    has_num_re = r".*[0-9].*"

    

    for word in text.split():

        if word.isnumeric():

            res.append("NUM")

        elif not (re.search(has_char_re, word) and re.search(has_num_re, word)):

            res.append(word)

        else:

            res.append("MIXNUM")

            

    return " ".join(res)





def preprocess_text(text):

    text = " ".join(text.split())

    text = text.replace("|||", " SEPAR ")

    text = tokenize(text)

    text = remove_stop_words(text)

    text = normalize_text(text)



    res = text



    return res





def preprocess_df(df):

    df.question = df.question.apply(preprocess_text)

    df.answer = df.answer.apply(preprocess_text)

    

    return df
%%time

proc_df = preprocess_df(df)
proc_df.tail()
proc_df = proc_df.fillna("")
from pyvi.ViPosTagger import postagging

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from scipy.sparse import hstack

import numpy as np

from sklearn.preprocessing import MinMaxScaler





def clean_text(text):

    res = []

    

    for word in text.split():

        if word not in ["NUM", "MIXNUM", "SEPAR"]:

            res.append(word)

            

    return " ".join(res)





def create_train_feature(df):

    ques = df.question.to_list()

    

    ans = df.answer.to_list()

    

    corpus = [q + " " + a for q, a in zip(ques, ans)]



    n_num = np.array([c.count("NUM") for c in corpus])

    n_mix = np.array([c.count("MIXNUM") for c in corpus])

    n_ques = np.array([q.count("SEPAR") + 1 for q in ques])

    n_ans = np.array([a.count("SEPAR") + 1 for a in ans])



    numeric_feature = np.vstack([n_num, n_mix, n_ques, n_ans]).T

    scaler = MinMaxScaler()

    numeric_feature = scaler.fit_transform(numeric_feature)

    

    

    corpus = [clean_text(c) for c in corpus]

    ques = [clean_text(q) for q in ques]

    corpus = [clean_text(a) for a in ans]



    ques_lower = [q.lower() for q in ques]

    ques_vectorizer = fit_tfidf(ques_lower)



    ans_lower = [a.lower() for a in ans]

    ans_vectorizer = fit_tfidf(ans_lower)



    pos_corpus = []

    for text in corpus:

        pos_text = " ".join(postagging(text)[1])

        pos_corpus.append(pos_text)

    pos_vectorizer = fit_countvec(pos_corpus)

    

    ques_feature = ques_vectorizer.transform(ques_lower)

    ans_feature = ans_vectorizer.transform(ans_lower)

    pos_feature = pos_vectorizer.transform(pos_corpus)

    

    return hstack((ques_feature, ans_feature, pos_feature, numeric_feature)), ques_vectorizer, ans_vectorizer, pos_vectorizer, scaler





def create_test_feature(df, ques_vectorizer, ans_vectorizer, pos_vectorizer, scaler):

    ques = df.question.to_list()

    

    ans = df.answer.to_list()

    

    corpus = [q + " " + a for q, a in zip(ques, ans)]



    n_num = np.array([c.count("NUM") for c in corpus])

    n_mix = np.array([c.count("MIXNUM") for c in corpus])

    n_ques = np.array([q.count("SEPAR") + 1 for q in ques])

    n_ans = np.array([a.count("SEPAR") + 1 for a in ans])

    

    numeric_feature = np.vstack([n_num, n_mix, n_ques, n_ans]).T

    numeric_feature = scaler.transform(numeric_feature)

    

    corpus = [clean_text(c) for c in corpus]

    ques = [clean_text(q) for q in ques]

    corpus = [clean_text(a) for a in ans]



    ques_lower = [q.lower() for q in ques]



    ans_lower = [a.lower() for a in ans]



    pos_corpus = []

    for text in corpus:

        pos_text = " ".join(postagging(text)[1])

        pos_corpus.append(pos_text)

    

    ques_feature = ques_vectorizer.transform(ques_lower)

    ans_feature = ans_vectorizer.transform(ans_lower)

    pos_feature = pos_vectorizer.transform(pos_corpus)

    

    return hstack((ques_feature, ans_feature, pos_feature, numeric_feature))



        

def fit_tfidf(corpus):

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)

    return vectorizer.fit(corpus)





def fit_countvec(corpus):

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, token_pattern="(?u)\\b\\w+\\b")

    return vectorizer.fit(corpus)
%%time

X, t, c, pos, s = create_train_feature(proc_df)
print(f"Có tổng cộng {X.shape[1]} features")

print(f"Shape X: {X.shape}")
y = np.array([0 if l == "N" else 1 for l in df.is_unsatisfied])
from sklearn.linear_model import LogisticRegression
params = {"C": 0.1,

          "penalty": "l1",

          "solver": "liblinear",

          "class_weight": "balanced",

          "random_state": 42, 

          "max_iter": 1000}



lg = LogisticRegression(**params)
%%time

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))



acc_score = []



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    X_ktrain = X.tocsr()[train_index, :]

    X_kvalid = X.tocsr()[valid_index, :]

    y_ktrain, y_kvalid = y[train_index], y[valid_index]

    

    lg.fit(X_ktrain, y_ktrain)

    acc_score.append(accuracy_score(y_kvalid, lg.predict(X_kvalid)))

    viz = plot_roc_curve(lg, X_kvalid, y_kvalid,

                         name=f"ROC fold {i + 1}",

                         alpha=0.3, lw=1, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)



ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",

        label="Chance", alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color="b",

        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,

                label=r"$\pm$ 1 std. dev.")



ax.set(xlim=[0, 1], ylim=[0, 1],

       title="ROC")

ax.legend(loc="lower right")

plt.show()
print(f"Mean accuracy: {np.mean(acc_score)} - Std accuracy: {np.std(acc_score)}")
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB(var_smoothing=1e-1)
%%time

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))



acc_score = []



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    X_ktrain = X.toarray()[train_index, :]

    X_kvalid = X.toarray()[valid_index, :]

    y_ktrain, y_kvalid = y[train_index], y[valid_index]

    

    nb.fit(X_ktrain, y_ktrain)

    acc_score.append(accuracy_score(y_kvalid, nb.predict(X_kvalid)))

    viz = plot_roc_curve(nb, X_kvalid, y_kvalid,

                         name=f"ROC fold {i + 1}",

                         alpha=0.3, lw=1, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)



ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",

        label="Chance", alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color="b",

        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,

                label=r"$\pm$ 1 std. dev.")



ax.set(xlim=[0, 1], ylim=[0, 1],

       title="ROC")

ax.legend(loc="lower right")

plt.show()
print(f"Mean accuracy: {np.mean(acc_score)} - Std accuracy: {np.std(acc_score)}")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, 

                            n_estimators=500, 

                            min_samples_split=0.02,

                            class_weight="balanced",

                            random_state=42)
%%time

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))



acc_score = []



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    X_ktrain = X.tocsr()[train_index, :]

    X_kvalid = X.tocsr()[valid_index, :]

    y_ktrain, y_kvalid = y[train_index], y[valid_index]

    

    rf.fit(X_ktrain, y_ktrain)

    acc_score.append(accuracy_score(y_kvalid, rf.predict(X_kvalid)))

    viz = plot_roc_curve(rf, X_kvalid, y_kvalid,

                         name=f"ROC fold {i + 1}",

                         alpha=0.3, lw=1, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)



ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",

        label="Chance", alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color="b",

        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,

                label=r"$\pm$ 1 std. dev.")



ax.set(xlim=[0, 1], ylim=[0, 1],

       title="ROC")

ax.legend(loc="lower right")

plt.show()
print(f"Mean accuracy: {np.mean(acc_score)} - Std accuracy: {np.std(acc_score)}")
from sklearn.ensemble import StackingClassifier
estimators = [

    ("lg", lg),

    ("nb", nb),

    ("rf", rf)

]

clf = StackingClassifier(

    estimators=estimators, 

    final_estimator=LogisticRegression(),

    cv=5

)
%%time

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))



acc_score = []



for i, (train_index, valid_index) in enumerate(skf.split(X, y)):

    X_ktrain = X.toarray()[train_index, :]

    X_kvalid = X.toarray()[valid_index, :]

    y_ktrain, y_kvalid = y[train_index], y[valid_index]

    

    clf.fit(X_ktrain, y_ktrain)

    acc_score.append(accuracy_score(y_kvalid, clf.predict(X_kvalid)))

    viz = plot_roc_curve(clf, X_kvalid, y_kvalid,

                         name=f"ROC fold {i + 1}",

                         alpha=0.3, lw=1, ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)

    aucs.append(viz.roc_auc)



ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",

        label="Chance", alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color="b",

        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.2,

                label=r"$\pm$ 1 std. dev.")



ax.set(xlim=[0, 1], ylim=[0, 1],

       title="ROC")

ax.legend(loc="lower right")

plt.show()
print(f"Mean accuracy: {np.mean(acc_score)} - Std accuracy: {np.std(acc_score)}")
from feature_extraction import create_test_feature





best_model = clf



submit_df = pd.read_csv("../input/isods-dataset/testing_data_sentiment.csv")

submit_df = preprocess_df(submit_df)



X_test = create_test_feature(submit_df, t, c, pos, s)

best_model.fit(X.toarray(), y)



y_pred = best_model.predict(X_test.toarray())

y_pred = ["N" if i == 0 else "Y" for i in y_pred]



submit = pd.DataFrame({"num": range(1, len(y_pred) + 1), "label": y_pred})

submit.to_csv("submit.csv", index=False)
import joblib

joblib.dump(clf, "stacking_model.joblib")

joblib.dump(t, "question_vertorizer.joblib")

joblib.dump(c, "answer_vertorizer.joblib")

joblib.dump(pos, "pos_vertorizer.joblib")

joblib.dump(s, "scaler.joblib")