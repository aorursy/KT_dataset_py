import numpy as np

import pandas as pd



from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



import nltk

from sklearn.feature_extraction.text import TfidfVectorizer



import gc

from tqdm.notebook import tqdm
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col=0, header=0)

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', index_col=0, header=0)



train_obs = list(df_train.index)

test_obs = list(df_test.index)



df_train['text'].values[0]
tokenizer_, lemmatizer_ = nltk.TreebankWordTokenizer(), nltk.stem.WordNetLemmatizer()

tokenize_lemmatize = lambda s: ' '.join([lemmatizer_.lemmatize(token.lower()) for token in tokenizer_.tokenize(s)])



lemmas_train = list(map(tokenize_lemmatize, df_train['text'].fillna('_na_').values))

lemmas_test = list(map(tokenize_lemmatize, df_test['text'].fillna('_na_').values))



lemmas_train[0]
vec = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_df=0.75, min_df=3, max_features=10_000)

# ngram_range=(1, 2) means that token may be not just a signle worde, but also a pair of consecutive words

# max_df=0.75 we don't consider tokens which appear in more than 75% tweets (e.g. articles "a", "the", pronouns "you", "we", "our" and so on)

# min_df=3 we don't consider tokens which appear in less than 3 tweets (e.g. typos and rare proper nouns )

# max_features=10_000 we use 10k most common tokens

vec.fit(lemmas_train + lemmas_test)



lemmas_train = vec.transform(lemmas_train)

lemmas_test = vec.transform(lemmas_test)



print(lemmas_train[0])

print(lemmas_train.shape)
y_train = df_train['target'].values
n_folds = 5

kf = StratifiedKFold(n_splits=n_folds, shuffle=True)



lr_train_probs = np.zeros(lemmas_train.shape[0], np.float32)

lr_test_probs = np.zeros(lemmas_test.shape[0], np.float32)



for train_index, valid_index in kf.split(lemmas_train, y_train):

    lr = LogisticRegression(C=1, solver='lbfgs')

    lr.fit(lemmas_train[train_index], y_train[train_index])

    lr_train_probs[valid_index] = lr.predict_proba(lemmas_train[valid_index])[:, 1]

    lr_test_probs += lr.predict_proba(lemmas_test)[:, 1]

    

lr_test_probs /= n_folds
bins = np.linspace(0, 1, num=101)



f1 = []

for eps in bins:

    train_labels = (lr_train_probs >= eps).astype(np.uint8)

    f1.append(f1_score(y_train, train_labels))

    

k = np.argmax(f1)

eps = bins[k]



f'Train metric: {f1[k]}'
test_labels = (lr_test_probs >= eps).astype(np.uint8)



sub = pd.DataFrame(test_labels.reshape(-1, 1), index=test_obs, columns=['target'])

sub.index.name = 'id'

sub.to_csv('submission.csv')



sub.head()