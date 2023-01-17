import numpy as np

import pandas as pd



from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from pathlib import Path



# ignore all future warnings

import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(31)
PATH = Path('../input')
!ls '{PATH}'
data = pd.read_csv(PATH/'haha_2019_train.csv')

data.shape
data.dtypes
size = data.shape[0]

train_size = int(size * 0.85)

valid_size = size - train_size



size, train_size, valid_size
indexes = np.random.permutation(size)

train_indexes = indexes[:(train_size-1)]

valid_indexes = indexes[train_size:]
train = data.loc[train_indexes]

valid = data.loc[valid_indexes]



train.shape, valid.shape
train.describe()
train.head()
train.tail()
train['funniness_average'].hist()
nan = train.loc[3]['funniness_average']

type(nan), nan
train[train['funniness_average']==nan].count()
train['funniness_average'].fillna(0.0, inplace=True)

valid['funniness_average'].fillna(0.0, inplace=True)
funniness_average = train['funniness_average']

mean, std = funniness_average.mean(), funniness_average.std()

funniness_average.min(), funniness_average.max(), funniness_average.mean(), funniness_average.std()
((train['funniness_average']-mean)/std).mean()
train['funniness_average_normalized'] = (train['funniness_average']-mean)/std

valid['funniness_average_normalized'] = (valid['funniness_average']-mean)/std
assert train['funniness_average_normalized'].mean() < 1e-3       # train mean should be 0

assert abs(1-train['funniness_average_normalized'].std()) < 1e-3 # train std dev should be 1

valid['funniness_average_normalized'].mean(), valid['funniness_average_normalized'].std()
train.head()
train['text'][0]
train['text'][4]
lens = train.text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist()
data['text'].fillna('', inplace=True)
train['is_humor'].unique(), train['is_humor'].nunique()
is_humor = train['is_humor']

is_humor.min(), is_humor.max(), is_humor.mean(), is_humor.std()
is_humor.hist()
votes_no = train['votes_no']

votes_no.min(), votes_no.max(), votes_no.mean(), votes_no.std()
train[train['votes_no']==1290]
votes_no.hist()
train[(train['is_humor']==0) & (train['votes_no']==0)]
train[(train['is_humor']==1) & (train['votes_no']>0)].T
labels_binary = ['is_humor']

labels_categorical = ['votes_no', 'votes_1', 'votes_2', 'votes_3', 'votes_4', 'votes_5']

labels_continuous = ['funniness_average_normalized']
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
train.loc[0].text, tokenize(train.loc[0].text)
n = train.shape[0]

vec = TfidfVectorizer(

                ngram_range=(1,2),       # The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

                tokenizer=tokenize,      # Override the string tokenization step while preserving the preprocessing and n-grams generation steps

                min_df=3,                # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.

                max_df=0.9,              # When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).

                strip_accents='unicode', # Remove accents and perform other character normalization during the preprocessing step

                use_idf=True,            # Enable inverse-document-frequency reweighting

                smooth_idf=True,         # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.

                sublinear_tf=True        # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

                )

train_term_doc = vec.fit_transform(train['text'])

valid_term_doc = vec.transform(valid['text'])
train_term_doc, valid_term_doc
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x = train_term_doc

valid_x = valid_term_doc
def get_logistic_model(y):

    y = y.values

    r = np.log(pr(1, y) / pr(0, y))

    m = LogisticRegression(C=4, dual=True) # LR with L2 regularization

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
def get_linear_model(y):

    y = y.values

    r = np.log(pr(1, y) / pr(0, y))

    m = LinearRegression() # LR with L2 regularization

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
m, r = get_logistic_model(train['votes_4'])

scores = m.predict(valid_x.multiply(r))

scores, np.unique(scores)
dict([(lbl, np.int64) for lbl in labels_categorical])
preds = np.zeros((len(valid), len(labels_binary)+len(labels_categorical)+len(labels_continuous)))



for i, j in enumerate(labels_binary):

    print(f'Fitting {j}')

    m, r = get_logistic_model(train[j])

    preds[:, i] = m.predict_proba(valid_x.multiply(r))[:, 1]



for i, j in enumerate(labels_categorical):

    print(f'Fitting {j}')

    m, r = get_logistic_model(train[j])

    preds[:, i+len(labels_binary)] = m.predict(valid_x.multiply(r))

    

for i, j in enumerate(labels_continuous):

    print(f'Fitting {j}')

    m, r = get_linear_model(train[j])

    preds[:, i+len(labels_binary)+len(labels_categorical)] = m.predict(valid_x.multiply(r))
preds = pd.DataFrame(preds, columns=dict([(lbl, np.float64) for lbl in labels_binary] + [(lbl, np.int64) for lbl in labels_categorical] + [(lbl, np.float64) for lbl in labels_continuous]))

for lbl in labels_categorical: preds[lbl] = preds[lbl].apply(int)

preds.head()
for i, j in enumerate(labels_binary):

    acc = ((preds[j].values>0.5).astype(int)==valid[j]).sum() / len(valid)

    print(f'Accuracy of predictions for {j} = {acc}')