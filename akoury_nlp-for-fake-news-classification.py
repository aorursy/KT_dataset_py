import re

import eli5

import spacy

import nltk as nl

import pandas as pd

from sklearn.base import clone

import matplotlib.pyplot as plt

from scipy.sparse import hstack

from nltk.corpus import stopwords

from ml_helper.helper import Helper

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from nltk.stem.snowball import SnowballStemmer

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from scikitplot.metrics import plot_confusion_matrix

from sklearn.preprocessing import FunctionTransformer

from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import accuracy_score as metric_scorer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

nl.download('stopwords')

%matplotlib inline
KEYS = {

    "SEED": 1,

    "DATA_PATH": "../input/fake_or_real_news.csv",

    "TARGET": "label",

    "METRIC": "accuracy",

    "TIMESERIES": False,

    "SPLITS": 3,

    "ESTIMATORS": 150,

    "ITERATIONS": 500,

}



hp = Helper(KEYS)
df = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])

train, test = train_test_split(df, test_size=0.20, random_state=KEYS["SEED"])
train.dtypes
train.head()
hp.missing_data(df)
train.iloc[10,2]
hp.target_distribution(train)
train["concat"] = train["title"] + train["text"]

test["concat"] = test["title"] + test["text"]

train.head()
count_vect = CountVectorizer()

base_train = count_vect.fit_transform(train["concat"])

print(base_train.shape)
tfidf_transformer = TfidfTransformer()

base_train = tfidf_transformer.fit_transform(base_train)

print(base_train.shape)
pcaed = TruncatedSVD(n_components=2).fit_transform(base_train)

pcaed = pd.concat([pd.DataFrame(pcaed).reset_index(drop=True), train["label"].reset_index(drop=True)], axis=1)

pcaed.head()
pctrue=pcaed[pcaed["label"]=="REAL"]

pcfake=pcaed[pcaed["label"]=="FAKE"]

plt.figure(figsize = (12,8))

plt.xlabel('Principal Component 1', fontsize = 15)

plt.ylabel('Principal Component 2', fontsize = 15)

plt.title('2 Component PCA on TF-IDF Representation', fontsize = 20)

plt.scatter(pctrue[0], pctrue[1], color="blue")

plt.scatter(pcfake[0], pcfake[1], color="red")

plt.legend(["REAL", "FAKE"])
basepipe = Pipeline([

    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True))

])

    



models = [

    {"name": "naive", "model": MultinomialNB()},

    {"name": "logistic_regression", "model": LogisticRegression(solver="lbfgs", max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"])},

    {"name": "svm", "model": SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=KEYS["SEED"])},

    {"name": "pac", "model":  PassiveAggressiveClassifier(max_iter=1000, random_state=KEYS["SEED"], tol=1e-3)},

]



all_scores = hp.pipeline(train[["concat", "label"]], models, basepipe, note="Base models")
def stemmer(df, stem = "snow"):

    if stem == "port":

        stemmer = PorterStemmer()

    else:

        stemmer = SnowballStemmer(language='english')

    df = df.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())

    return df
train.iloc[3:4]["concat"]
stemmer(train["concat"].iloc[3:4])
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

doc = nlp(train['concat'][3])

print(doc.text[:50])

print('----------------------------------------------------')

for token in doc[:5]:

    print(f'Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}')
def tokenizer(text):

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

    return [token.lemma_.lower().strip() + token.pos_ for token in nlp(text)]
def strict_tokenizer(text):

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

    return [token.lemma_.lower().strip() + token.pos_ for token in nlp(text)

        if 

            not token.is_stop and not nlp.vocab[token.lemma_].is_stop

            and not token.is_punct

            and not token.is_digit

    ]
snow_pipe = Pipeline([

    ('snow_stem', FunctionTransformer(stemmer, validate=False)),

    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)),

])
port_pipe = Pipeline([

    ('port_stem', FunctionTransformer(stemmer, kw_args={"stem": "port"}, validate=False)),

    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)),

])
lemm_pipe = Pipeline([

    ('lemma_vect', TfidfVectorizer(analyzer = 'word', max_df=0.99, min_df=0.01, ngram_range=(1,2), tokenizer=tokenizer))

])
strict_lemm_pipe = Pipeline([

    ('strict_lemma_vect', TfidfVectorizer(analyzer = 'word', max_df=0.99, min_df=0.01, ngram_range=(1,2), tokenizer=strict_tokenizer))

])
models = [

    {"name": "logistic_regression", "model": LogisticRegression(solver="lbfgs", max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"])},

    {"name": "pac", "model":  PassiveAggressiveClassifier(max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"], tol=1e-3)},

]
all_scores = hp.pipeline(train[["concat", "label"]], models, snow_pipe, all_scores=all_scores, quiet = True)

all_scores = hp.pipeline(train[["concat", "label"]], models, port_pipe, all_scores=all_scores, quiet = True)

all_scores = hp.pipeline(train[["concat", "label"]], models, lemm_pipe, all_scores=all_scores, quiet = True)

all_scores = hp.pipeline(train[["concat", "label"]], models, strict_lemm_pipe, all_scores=all_scores)
hp.plot_models(all_scores)
hp.show_scores(all_scores, top=True)
grid = {

    "pac__C": [1.0, 10.0],

    "pac__tol": [1e-2, 1e-3],

    "pac__max_iter": [500, 1000],

}



final_scores, pipe = hp.cross_val(train[["concat", "label"]], model=clone(hp.top_pipeline(all_scores)), grid=grid)

final_scores
print(pipe.best_params_)

final_pipe = pipe.best_estimator_
eli5.show_weights(final_pipe, top=30, target_names=train.label)
test.concat.iloc[2]
eli5.show_prediction(final_pipe.named_steps['pac'], test.concat.iloc[2], vec = final_pipe.named_steps['lemma_vect'], top=30, target_names=train.label)
predictions = final_pipe.predict(test["concat"])

metric_scorer(test["label"], predictions)
print(classification_report(test["label"], predictions))
plot_confusion_matrix(test["label"], predictions)