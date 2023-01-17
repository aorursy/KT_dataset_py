!pip -q install pyspellchecker optuna
import os

import optuna

import pandas as pd

import numpy as np

import random

import re

from scipy import sparse

from spellchecker import SpellChecker

import string

import warnings



from IPython.display import HTML



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.utils.validation import check_X_y, check_is_fitted



from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
optuna.logging.set_verbosity(0)

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
SEED = 42

NUM_SPLITS = 10

NUM_TRIALS = 100
# setting seed

os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
_ = plt.figure(figsize=(6, 4))

sns.countplot(x="target", data=train)

_ = plt.title("Target Distribution")
test["target"] = -1

df = pd.concat([train, test])
print("NaN Distribution\n")

for col in df.columns:

    print(f"{col}: {((df[col].isna().sum()/df.shape[0])*100):.2f}")
HTML('<div style="position:relative;height:0;padding-bottom:56.25%"><iframe src="https://youtu.be/3w92peJtYNQ" width="640" height="360" frameborder="0" style="position:absolute;width:100%;height:100%;left:0" allowfullscreen></iframe></div>')
# loading augmented training data

train = pd.read_csv("../input/twitter-sentiment-easy-data-augmentation/train_augmented.csv")
_ = plt.figure(figsize=(6, 4))

sns.countplot(x="target", data=train)

_ = plt.title("Augmented Data - Target Distribution")
df = pd.concat([train, test])
train.shape, test.shape, df.shape
df["text"] = df["text"].str.lower()
PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



df["text"] = df["text"].apply(lambda text: remove_punctuation(text))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



df["text"] = df["text"].apply(lambda text: remove_emoji(text))
def remove_urls(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    try:

        return url_pattern.sub(r'', text)

    except:

        print(text)

    

df["text"] = df["text"].apply(lambda text: remove_urls(text))
def remove_html(text):

    html_pattern = re.compile('<.*?>')

    return html_pattern.sub(r'', text)



df["text"] = df["text"].apply(lambda text: remove_html(text))
with open("../input/slangtext/slang.txt", "r") as file:

    chat_words_str = file.read()



chat_words_map_dict = {}

chat_words_list = []

for line in chat_words_str.split("\n"):

    if line != "" and "=" in line:

        cw = line.split("=")[0]

        cw_expanded = line.split("=")[1]

        chat_words_list.append(cw)

        chat_words_map_dict[cw] = cw_expanded

chat_words_list = set(chat_words_list)
def chat_words_conversion(text):

    new_text = []

    for w in text.split():

        if w.upper() in chat_words_list:

            new_text.append(chat_words_map_dict[w.upper()])

        else:

            new_text.append(w)

    return " ".join(new_text)



df["text"] = df["text"].apply(lambda text: chat_words_conversion(text))
spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



df["text"] = df["text"].apply(lambda text: chat_words_conversion(text))
class NBSVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, max_iter=100, dual=False, n_jobs=1):

        self.C = C

        self.dual = dual

        self.n_jobs = n_jobs

        self.max_iter = max_iter



    def predict(self, x):

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict(x.multiply(self._r))



    def fit(self, x, y):

        y = y.values

        x, y = check_X_y(x, y, accept_sparse=True)



        def pr(x, y_i, y):

            p = x[y==y_i].sum(0)

            return (p+1) / ((y==y_i).sum()+1)



        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))

        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(C=self.C, dual=self.dual, 

                                       max_iter=self.max_iter, 

                                       n_jobs=self.n_jobs).fit(x_nb, y)

        return self
X_train, X_valid, y_train, y_valid = train_test_split(train["text"], train["target"],

                                                      test_size=0.2, random_state=SEED,

                                                      stratify=train["target"])
vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, 

                      strip_accents='unicode', use_idf=1,

                      smooth_idf=1, sublinear_tf=1)
def objective(trial):

    C = trial.suggest_float(name="C", low=1e-3, high=1e3, log=True)

    max_iter = trial.suggest_discrete_uniform(name="max_iter", low=50, high=500, q=50)

    nbsvm = NBSVMClassifier(C=C, max_iter=max_iter)

    

    train_term_doc = vec.fit_transform(X_train)

    valid_term_doc = vec.transform(X_valid)

    nbsvm.fit(train_term_doc, y_train)

    

    preds = nbsvm.predict(valid_term_doc)

    preds[preds>=0.5] = 1

    preds[preds<0.5] = 0

    

    acc = accuracy_score(y_valid, preds)

    return acc
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
print(f"Best Value: {study.best_trial.value}")

print(f"Best Params: {study.best_params}")
kwargs = study.best_params
train = df[df['target']!=-1]

test = df[df['target']==-1]
def print_metrics(y_true, y_pred):

    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

    print(f"MCC: {matthews_corrcoef(y_true, y_pred)}")

    print(f"F1: {f1_score(y_true, y_pred)}\n")
final_preds = np.zeros((len(test)))

kfold = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)



for fold, (train_index, valid_index) in enumerate(kfold.split(train["text"], train["target"])):

    print("*"*60)

    print("*"+" "*26+f"FOLD {fold+1}"+" "*26+"*")

    print("*"*60, end="\n")    

    

    X_train = train.iloc[train_index, :].reset_index(drop=True)

    X_valid = train.iloc[valid_index, :].reset_index(drop=True)

    

    y_train = X_train['target']

    y_valid = X_valid['target']

    

    train_term_doc = vec.fit_transform(X_train["text"])

    valid_term_doc = vec.transform(X_valid["text"])

    test_term_doc = vec.transform(test["text"])

    

    # using best hyperparameters selected above

    nbsvm = NBSVMClassifier(**kwargs)

    nbsvm.fit(train_term_doc, y_train)

    

    valid_preds = nbsvm.predict(valid_term_doc)

    print_metrics(y_valid, valid_preds)

    

    test_preds = nbsvm.predict(test_term_doc)

    final_preds += test_preds
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission["target"] = final_preds/NUM_SPLITS

submission["target"] = submission["target"].apply(lambda x: 1 if x>=0.5 else 0)
submission.to_csv("submission.csv", index=False)