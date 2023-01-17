import re
import json
import string
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import gc
import pickle
from tqdm import tqdm
tqdm.pandas()

%matplotlib inline

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

import time
from contextlib import contextmanager

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
@contextmanager
def timer(task_name="timer"):
    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    print("----> {} started".format(task_name))
    t0 = time.time()
    yield
    print("----> {} done in {:.0f} seconds".format(task_name, time.time() - t0))
with timer("reading_data"):
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    df = pd.concat([train_df ,test_df]).reset_index(drop=True)

    y_train = train_df["target"].values

    print('Total:', df.shape)
    print('Train:', train_df.shape)
    print('Test:', test_df.shape)
    print("Number of texts: ", df.shape[0])
print(train_df['target'].value_counts())
sns.countplot(train_df['target'])
plt.show()
import psutil
from multiprocessing import Pool

num_cores = psutil.cpu_count()  # number of cores on your machine
num_partitions = num_cores  # number of partitions to split dataframe

print('number of cores:', num_cores)
def df_parallelize_run(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def text_cleaning(text):
    text = clean_text(text)
    text = clean_numbers(text)
    text = replace_typical_misspell(text)
    return text

def text_clean_wrapper(df):
    df["question_text"] = df["question_text"].progress_apply(text_cleaning)
    return df
with timer("basic_feature_engineering"):
    from nltk.corpus import stopwords
    STOPWORDS = list(set(stopwords.words('english')))

    def count_regexp_occ(regexp="", text=None):
        """ Simple way to get the number of occurence of a regex"""
        return len(re.findall(regexp, text))

    # Count number of \n
    df["ant_slash_n"] = df['question_text'].progress_apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df['question_text'].progress_apply(lambda x: len(x.split()))
    df["raw_char_len"] = df['question_text'].progress_apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df['question_text'].progress_apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    df["nb_title"] = df["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    # stopwords count
    df["num_stopwords"] = df["question_text"].progress_apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# text cleaning
with timer("text_cleaning"):
    df = df_parallelize_run(df, text_clean_wrapper)
train_df = df.loc[:train_df.shape[0] - 1, :]
test_df = df.loc[train_df.shape[0]:, :]
del test_df['target']; del df
gc.collect()

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print('Train:', train_df.shape)
print('Test:', test_df.shape)
handcraft_feature_names = [f_ for f_ in test_df.columns if f_ not in ["question_text", "qid"]]
train_handcraft_features = train_df[handcraft_feature_names]
test_handcraft_features = test_df[handcraft_feature_names]
print('handcraft feature count:', len(handcraft_feature_names))
all_text = pd.concat([train_df['question_text'], test_df['question_text']], axis =0)
with timer("word_vectorizer"):
    word_vectorizer = TfidfVectorizer(
                    ngram_range=(1,4),
                    token_pattern=r'\w{1,}',
                    min_df=3,
                    max_df=0.9,
                    strip_accents='unicode',
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True,
                    max_features=100000
                    )
    
    word_vectorizer.fit(all_text)
    train_word_tfidf_features  = word_vectorizer.transform(train_df['question_text'])
    test_word_tfidf_features  = word_vectorizer.transform(test_df['question_text'])
def prepare_for_char_n_gram(text):
    """
    The word hashing method described here aim to reduce the
    dimensionality of the bag-of-words term vectors. It is based on
    letter n-gram, and is a new method developed especially for our
    task. Given a word (e.g. good), we first add word starting and
    ending marks to the word (e.g. #good#). Then, we break the word
    into letter n-grams (e.g. letter trigrams: #go, goo, ood, od#).
    Finally, the word is represented using a vector of letter n-grams. 
    """
    text = re.sub(" ", "# #", text)  # Replace space
    text = "#" + text + "#"  # add leading and trailing #
    return text

train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: prepare_for_char_n_gram(x))
test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: prepare_for_char_n_gram(x))
with timer("char_vectorizer"):
    def char_analyzer(text):
        """
        This is used to split strings in small lots
        Word Hashing: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf
        """
        tokens = text.split()
        return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

    char_vectorizer = TfidfVectorizer(
                        ngram_range=(1,1),
                        tokenizer=char_analyzer,
                        min_df=3,
                        max_df=0.9,
                        strip_accents='unicode',
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True,
                        max_features=50000
                        )

    char_vectorizer.fit(all_text)
    train_char_tfidf_features = char_vectorizer.transform(train_df['question_text'])
    test_char_tfidf_features = char_vectorizer.transform(test_df['question_text'])
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class NBTransformer(BaseEstimator, TransformerMixin):
    """
    https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
    """
    def __init__(self, alpha=1):
        self.r = None
        self.alpha = alpha

    def fit(self, X, y):
        # store smoothed log count ratio
        p = self.alpha + X[y==1].sum(0)
        q = self.alpha + X[y==0].sum(0)
        self.r = csr_matrix(np.log(
            (p / (self.alpha + (y==1).sum())) /
            (q / (self.alpha + (y==0).sum()))
        ))
        return self

    def transform(self, X, y=None):
        return X.multiply(self.r)
with timer("transform_Naive_Bayes"):
    # transform to Naive Bayes feature
    nb_transformer = NBTransformer(alpha=1).fit(train_word_tfidf_features, y_train)
    train_word_tfidf_features = nb_transformer.transform(train_word_tfidf_features)
    test_word_tfidf_features = nb_transformer.transform(test_word_tfidf_features)
    
    nb_transformer = NBTransformer(alpha=1).fit(train_char_tfidf_features, y_train)
    train_char_tfidf_features = nb_transformer.transform(train_char_tfidf_features)
    test_char_tfidf_features = nb_transformer.transform(test_char_tfidf_features)
with timer("feature_concate"):
    feature_names = word_vectorizer.get_feature_names() + char_vectorizer.get_feature_names() + handcraft_feature_names
    del all_text; gc.collect()

    X_train = hstack(
            [
                train_word_tfidf_features,
                train_char_tfidf_features,
                train_handcraft_features
            ]
        ).tocsr()

    del train_word_tfidf_features; del train_char_tfidf_features; del train_handcraft_features
    gc.collect()

    X_test = hstack(
        [
            test_word_tfidf_features,
            test_char_tfidf_features,
            test_handcraft_features
        ]
    ).tocsr()
    del test_word_tfidf_features; del test_char_tfidf_features; del test_handcraft_features
    gc.collect()
print('Train:', X_train.shape)
print('Test:', X_test.shape)
print('feature count:', len(feature_names))
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

param = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "num_threads": -1,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_split_gain": .1,
        "reg_alpha": .1,
        
        "scale_pos_weight": 15,
    }
with timer("run-lightgbm-out-of-fold"):
    n_folds = 5
    sk_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)

    oof = np.zeros(X_train.shape[0])
    predictions = np.zeros(X_test.shape[0])
    feature_importance_df = pd.DataFrame()

    train_aucs = []
    valid_aucs = []
    for fold_, (trn_idx, val_idx) in enumerate(sk_folds.split(X_train, y_train)):
        with timer("run fold {}/{}".format(fold_ + 1, n_folds)):
            trn_data = lgb.Dataset(X_train[trn_idx], label=y_train[trn_idx])
            val_data = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, 
                            valid_sets = [trn_data, val_data], 
                            verbose_eval=100, early_stopping_rounds = 200)
            oof[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
            predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / sk_folds.n_splits

            train_aucs.append(clf.best_score['training']['auc'])
            valid_aucs.append(clf.best_score['valid_1']['auc'])

            # 当前 fold 训练的 feature importance
            fold_importance_df = pd.DataFrame()
#             fold_importance_df["feature"] = used_features
            fold_importance_df["feature"] = feature_names
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = fold_ + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('cv_auc')
    print('-'*50)
    print('Mean train-auc: {:<8.5f}, valid-auc: {:<8.5f}'.format(np.mean(train_aucs), np.mean(valid_aucs)))
    print('-'*50)
def threshold_searching(y_true, y_proba, verbose=True):
    from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    def threshold_search(y_true, y_proba):
        precision , recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001) 
        F = 2 / (1/precision + 1/recall)
        best_score = np.max(F)
        best_th = thresholds[np.argmax(F)]
        return best_th 


    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

    scores = []
    ths = []
    for train_index, test_index in rkf.split(y_true, y_true):
        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        # determine best threshold on 'train' part 
        best_threshold = threshold_search(y_true_train, y_prob_train)

        # use this threshold on 'test' part for score 
        sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))
        scores.append(sc)
        ths.append(best_threshold)

    best_th = np.mean(ths)
    score = np.mean(scores)

    if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score,5)}')

    return best_th, score
with timer("search-threshold"):
    best_threshold, cv_f1_score = threshold_searching(y_train, oof)
pred_test_y = (predictions > best_threshold).astype(int)
sub = test_df[['qid']]
sub['prediction'] = pred_test_y
sub.to_csv("submission.csv", index=False)
