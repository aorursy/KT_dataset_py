import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))

# running our benchmark code in this kernel lead to memory errors, so 
# we do a slightly less memory intensive procedure if this is True, 
# set this as False if you are running on a computer with a lot of RAM
# it should be possible to use less memory in this kernel using generators
# rather than storing everything in RAM, but we won't explore that here
RUNNING_KAGGLE_KERNEL = True 
rspct_df = pd.read_csv('../input/rspct.tsv', sep='\t')
info_df  = pd.read_csv('../input/subreddit_info.csv')
rspct_df.head(5)
# note that info_df has information on subreddits that are not in data, 
# we filter them out here

info_df = info_df[info_df.in_data].reset_index()
info_df.head(5)
# we join the title and selftext into one field

def join_text(row):
    if RUNNING_KAGGLE_KERNEL:
        return row['title'][:100] + " " + row['selftext'][:512]
    else:
        return row['title'] + " " + row['selftext']

rspct_df['text'] = rspct_df[['title', 'selftext']].apply(join_text, axis=1)
# take the last 20% as a test set - N.B data is already randomly shuffled,
# and last 20% is a stratified split (equal proportions of subreddits)

train_split_index = int(len(rspct_df) * 0.8)

train_df, test_df = rspct_df[:train_split_index], rspct_df[train_split_index:]
X_train , X_test  = train_df.text, test_df.text
y_train, y_test   = train_df.subreddit, test_df.subreddit
from sklearn.preprocessing import LabelEncoder

# label encode y

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test  = le.transform(y_test)

y_train[:5]
from sklearn.feature_extraction.text import TfidfVectorizer

# extract features from text using bag-of-words (single words + bigrams)
# use tfidf weighting (helps a little for Naive Bayes in general)
# note : you can do better than this by extracting more features, then 
# doing feature selection, but not enough memory on this kernel!

print('this cell will take about 10 minutes to run')

NUM_FEATURES = 30000 if RUNNING_KAGGLE_KERNEL else 100000

tf_idf_vectorizer = TfidfVectorizer(max_features = NUM_FEATURES,
                                min_df=5,
                                ngram_range=(1,2),
                                stop_words=None,
                                token_pattern='(?u)\\b\\w+\\b',
                            )

X_train = tf_idf_vectorizer.fit_transform(X_train)
X_test  = tf_idf_vectorizer.transform(X_test)

from sklearn.feature_selection import chi2, SelectKBest

# if we have more memory, select top 100000 features and select good features
if not RUNNING_KAGGLE_KERNEL:
    chi2_selector = SelectKBest(chi2, 30000)

    chi2_selector.fit(X_train, y_train) 

    X_train = chi2_selector.transform(X_train)
    X_test  = chi2_selector.transform(X_test)

X_train.shape, X_test.shape
from sklearn.naive_bayes import MultinomialNB

# train a naive bayes model, get predictions

nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)

y_pred_proba = nb_model.predict_proba(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
# we use precision-at-k metrics to evaluate performance
# (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)

def precision_at_k(y_true, y_pred, k=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.argsort(y_pred, axis=1)
    y_pred = y_pred[:, ::-1][:, :k]
    arr = [y in s for y, s in zip(y_true, y_pred)]
    return np.mean(arr)

print('precision@1 =', np.mean(y_test == y_pred))
print('precision@3 =', precision_at_k(y_test, y_pred_proba, 3))
print('precision@5 =', precision_at_k(y_test, y_pred_proba, 5))

# RUNNING_KAGGLE_KERNEL == True
# precision@1 = 0.610528134254689
# precision@3 = 0.7573692003948668
# precision@5 = 0.8067670286278381

# RUNNING_KAGGLE_KERNEL == False
# precision@1 = 0.7292102665350444
# precision@3 = 0.8512240868706812
# precision@5 = 0.8861500493583415
