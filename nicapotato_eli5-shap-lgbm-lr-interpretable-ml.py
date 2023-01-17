# from IPython.display import HTML

# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# The raw code for this IPython notebook is by default hidden for easier reading.
# To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# HYPER PARAMS
max_boosting_rounds = 5500

import time
notebookstart= time.time()

import datetime
import pandas as pd
import numpy as np
import random
import time

# Viz
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from wordcloud import WordCloud
import missingno as mn
from yellowbrick.text import TSNEVisualizer

# Hide Warnings
Warning = True
if Warning is False:
    import warnings
    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

# Modeling..
import eli5
import lightgbm as lgb
import shap
shap.initjs()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import scikitplot as skplt
from sklearn import preprocessing

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

np.random.seed(2018)

from contextlib import contextmanager
import re
import string
import gc

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
# Data Visualization
def cloud(text, title, size = (10,7)):
    # Processing Text
    wordcloud = WordCloud(width=800, height=400,
                          collocations=False
                         ).generate(" ".join(text))
    
    # Output Visualization
    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=25,color='w')
    plt.tight_layout(pad=0)
    plt.show()
train = pd.read_csv("../input/train.csv", index_col= 'qid')#.sample(50000)
test = pd.read_csv("../input/test.csv", index_col= 'qid')#.sample(5000)
testdex = test.index

target_names = ["Sincere","Insincere"]
y = train['target'].copy()
print(train.shape)
train.head()
print("Class Balance..")
train.target.value_counts(normalize=True)
for i,name in [(0,"Sincere"),(1,"Insincere")]:
     cloud(train.loc[train.target == i,"question_text"].str.title(), title="{} WordCloud".format(name), size=[8,5])
test['target'] = np.nan
all_text = pd.concat([train['question_text'],test['question_text']], axis =0)

word_vect = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)

with timer("Word Grams TFIDF"):
    word_vect.fit(all_text)
    X  = word_vect.transform(train['question_text'])
    testing  = word_vect.transform(test['question_text'])
# Train Test Split
X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=23, stratify=y)
# Create the visualizer and draw the vectors
plt.figure(figsize = [15,9])
tsne = TSNEVisualizer()
n = 20000
tsne.fit(X[:n], train.target[:n].map({1: target_names[1],0:target_names[0]}))
tsne.poof()
# Fit Model
model = LogisticRegression(solver = 'sag')
model.fit(X_train, y_train)

# Predict..
valid_logistic_pred = model.predict(X_valid)
train_logistic_pred = model.predict(X_train)
valid_logistic_pred_proba = model.predict_proba(X_valid)
valid_logistic_pred_proba = [x[1] for x in valid_logistic_pred_proba]
print("Train Set Accuracy: {}".format(metrics.accuracy_score(train_logistic_pred, y_train)))
print("Train Set ROC: {}".format(metrics.roc_auc_score(train_logistic_pred, y_train)))
print("Train Set F1 Score: {}\n".format(metrics.f1_score(train_logistic_pred, y_train)))

print("Validation Set Accuracy: {}".format(metrics.accuracy_score(valid_logistic_pred, y_valid)))
print("Validation Set ROC: {}".format(metrics.roc_auc_score(valid_logistic_pred, y_valid)))
print("Validation Set F1 Score: {}\n".format(metrics.f1_score(valid_logistic_pred, y_valid)))

print(metrics.classification_report(valid_logistic_pred, y_valid))

# Confusion Matrix
skplt.metrics.plot_confusion_matrix(valid_logistic_pred, y_valid)
plt.show()
eli5.show_weights(model, vec = word_vect, top=(50,50),
                  target_names=target_names)
def eli5_plotter(df, n = 5):
    for iteration in range(n):
        samp = random.randint(1,df.shape[0]-1)
        print("Ground Truth: {} \nPredicted: {}".format(
            pd.Series(df.target.iloc[samp]).map({0:'Sincere', 1: 'Insincere'})[0],
            pd.Series(df.predicted.iloc[samp]).map({0:'Sincere', 1: 'Insincere'})[0]))
        display(eli5.show_prediction(model, df.question_text.iloc[samp], vec=word_vect,
                             target_names=target_names))
        
# Prepare Validation Set
raw_valid = train.loc[train.index.isin(y_valid.index), :]
raw_valid['predicted'] = valid_logistic_pred
raw_valid['predicted_proba'] = valid_logistic_pred_proba

raw_valid['wrong_degree'] = abs(raw_valid['predicted_proba'] - raw_valid['target'])
temp = raw_valid.loc[raw_valid.sort_values(by='wrong_degree', ascending = False).index[:20], ["question_text", "target", "predicted"]]
eli5_plotter(temp, n=16)
eli5_plotter(temp, n=30)
temp = raw_valid.loc[raw_valid.question_text.str.contains("(?i)trump"),
                     ["question_text", "target", "predicted"]]
eli5_plotter(temp, n=10)
# submit_df = pd.DataFrame({"qid": test.index, "prediction": model.predict(testing).astype(int)})
# submit_df.to_csv("submission.csv", index=False)
# print(submit_df.head())
# del submit_df
###########################################################################################
### Upvote this :) - https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram ####
###########################################################################################

# The better written the code, the easier the copy pasta

# Contraction replacement patterns
cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))

def get_indicators_and_clean_comments(df, text_var):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df[text_var].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df[text_var].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df[text_var].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df[text_var].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df[text_var].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df[text_var].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df[text_var].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df[text_var].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df[text_var].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df[text_var].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df[text_var].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df[text_var].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df[text_var].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df[text_var].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df[text_var].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df[text_var].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df[text_var].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df[text_var].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    # Now clean comments
    df["clean_comment"] = df[text_var].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))
    
def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]

all_text = pd.concat([train['question_text'],test['question_text']], axis =0)

word_vect = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)

char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 1),
            max_features=50000)

with timer("Word Grams TFIDF"):
    word_vect.fit(all_text)
    train_word_features  = word_vect.transform(train['question_text'])
    test_word_features  = word_vect.transform(test['question_text'])

with timer("Character Grams TFIDF"):
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train['question_text'])
    test_char_features = char_vectorizer.transform(test['question_text'])

with timer("Performing basic NLP"):
    get_indicators_and_clean_comments(train, 'question_text')
    get_indicators_and_clean_comments(test,  'question_text')
    
    num_features = [f_ for f_ in train.columns
                if f_ not in ["question_text", "clean_comment", "remaining_chars",
                              'has_ip_address', 'target']]
    
# Get Sparse Matrix Feature Names..
feature_names = word_vect.get_feature_names() + char_vectorizer.get_feature_names() + num_features
del all_text; gc.collect()

with timer("Sparse Combine"):
    X = hstack(
        [
            train_char_features,
            train_word_features,
            train[num_features]
        ]
    ).tocsr()

    del train_char_features
    gc.collect()

    testing = hstack(
        [
            test_char_features,
            test_word_features,
            test[num_features]
        ]
    ).tocsr()
    del test_char_features; gc.collect()
# Create the visualizer and draw the vectors
plt.figure(figsize = [15,9])
tsne = TSNEVisualizer()
n = 20000
tsne.fit(X[:n], train.target[:n].map({1: target_names[1],0:target_names[0]}))
tsne.poof()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=23, stratify=y)

print("Light Gradient Boosting Classifier: ")
lgbm_params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_split_gain": .1,
        "reg_alpha": .1
    }

modelstart= time.time()
# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=feature_names)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=feature_names)

# Go Go Go
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round= max_boosting_rounds,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=150,
    verbose_eval=100
)

del lgtrain, lgvalid ;  gc.collect();
valid_pred = lgb_clf.predict(X_valid)
_thresh = []
for thresh in np.arange(0.1, 0.501, 0.01):
    _thresh.append([thresh, metrics.f1_score(y_valid, (valid_pred>thresh).astype(int))])
#     print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_valid, (valid_pred>thresh).astype(int))))

_thresh = np.array(_thresh)
best_id = _thresh[:,1].argmax()
best_thresh = _thresh[best_id][0]
print("Best Threshold: {}\nF1 Score: {}".format(best_thresh, _thresh[best_id][1]))
allmodelstart= time.time()
# Run Model with different Seeds
multi_seed_pred = dict()
all_feature_importance_df  = pd.DataFrame()

optimal_rounds = lgb_clf.best_iteration

lgtrain = lgb.Dataset(X, y, feature_name=feature_names)

all_seeds = [27,22]
for seeds_x in all_seeds:
    modelstart= time.time()
    print("Seed: ", seeds_x,)
    # Go Go Go
    lgbm_params["seed"] = seeds_x
    lgb_seed_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round = optimal_rounds + 1,
        verbose_eval=200)

    # Feature Importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feature_names
    fold_importance_df["importance"] = lgb_seed_clf.feature_importance()
    all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

    multi_seed_pred[seeds_x] =  list(lgb_seed_clf.predict(testing))
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    print("###########################################################################################")
    del lgb_seed_clf
    
del lgtrain, X
cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index
best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(8,10))
sns.barplot(x="importance", y="feature", 
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
print("All Model Runtime: %0.2f Minutes"%((time.time() - allmodelstart)/60))

# To DataFrame
sub_preds = pd.DataFrame.from_dict(multi_seed_pred).replace(0,0.000001)
del multi_seed_pred; gc.collect();

# Correlation Plot
f, ax = plt.subplots(figsize=[8,6])
sns.heatmap(sub_preds.corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot for Seed Diversified Models")
plt.show()
# Take Mean over Seed prediction
target_var = 'prediction'
mean_sub = sub_preds.mean(axis=1).rename(target_var)
mean_sub = (mean_sub > best_thresh).astype(int)
mean_sub.index = testdex

# Submit
mean_sub.to_csv('submission.csv',index = True, header=True)
print(mean_sub.value_counts(normalize=True))
mean_sub.head()
non_sparse = pd.DataFrame(X_valid[:400].toarray(), columns = feature_names)
print(non_sparse.shape)
explainer = shap.TreeExplainer(lgb_clf)
shap_values = explainer.shap_values(non_sparse)
# summarize the effects of all the features
shap.summary_plot(shap_values, non_sparse)
# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], non_sparse.iloc[0,:])
# valid_data = train.loc[train.index.isin(y_valid.index), :]

# # visualize the first prediction's explanation
# for iteration in range(15):
#     samp = random.randint(1,non_sparse.shape[0])
#     print("Real Label: {}".format(valid_data.target.iloc[samp]))
#     display(valid_data.question_text.iloc[samp])
#     display(shap.force_plot(explainer.expected_value, shap_values[samp,:], non_sparse.iloc[samp,:]))
shap.dependence_plot("trump", shap_values, non_sparse)
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))