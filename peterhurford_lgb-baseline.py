import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
import nltk
from nltk.corpus import stopwords
import string

eng_stopwords = set(stopwords.words("english"))

## Number of words in the text ##
train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))
test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))
test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train["num_chars"] = train["question_text"].apply(lambda x: len(str(x)))
test["num_chars"] = test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["num_words_upper"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["num_words_upper"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train["num_words_title"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test["num_words_title"] = test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test["mean_word_len"] = test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_text = train['question_text']
test_text = test['question_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=5000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
eng_features = ['num_words', 'num_unique_words', 'num_chars', 
                'num_stopwords', 'num_punctuations', 'num_words_upper', 
                'num_words_title', 'mean_word_len']
train_ = train[eng_features]
train_.head()
from scipy.sparse import hstack, csr_matrix
train_ = hstack((csr_matrix(train_), train_word_features))
print(train_.shape)
test_ = test[eng_features]
test_ = hstack((csr_matrix(test_), test_word_features))
print(test_.shape)
from sklearn.model_selection import train_test_split
y = train['target']
X_tr, X_va, y_tr, y_va = train_test_split(train_, y, test_size=0.2, random_state=42)
print(X_tr.shape, X_va.shape)
y_va.value_counts()
%%time
import lightgbm as lgb

from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

params = {'application': 'binary',
          'metric': 'binary_logloss',
          'learning_rate': 0.05,   
          'max_depth': 9,
          'num_leaves': 100,
          'verbosity': -1,
          'data_random_seed': 3,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.4,
          'nthread': 16,
          'lambda_l1': 1,
          'lambda_l2': 1,
          'num_rounds': 2700,
          'verbose_eval': 100}

d_train = lgb.Dataset(X_tr, label=y_tr.values)
d_valid = lgb.Dataset(X_va, label=y_va.values)
print('Train LGB')
num_rounds = params.pop('num_rounds')
verbose_eval = params.pop('verbose_eval')
model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=num_rounds,
                  valid_sets=[d_train, d_valid],
                  verbose_eval=verbose_eval,
                  valid_names=['train', 'val'],
                  feval=lgb_f1_score)
print('Predict')
pred_test_va = model.predict(X_va)
%%time
best_threshold = 0.01
best_score = 0.0
for threshold in range(1, 100):
    threshold = threshold / 100
    score = f1_score(y_va, pred_test_va > threshold)
    if score > best_score:
        best_threshold = threshold
        best_score = score
print(0.5, f1_score(y_va, pred_test_va > 0.5))
print(best_threshold, best_score)
# 0.24 0.5918758665447358
%%time
pred_test_y = model.predict(test_)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": (pred_test_y > best_threshold).astype(np.int)})
submit_df.head()
submit_df['prediction'].value_counts()
submit_df.to_csv("submission.csv", index=False)