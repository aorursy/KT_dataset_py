import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_val_predict

import csv
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')
train.head()
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist();
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)

train.describe()
# add label col 

# https://stackoverflow.com/questions/44464280/mapping-one-hot-encoded-target-values-to-proper-label-names

new_label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','none']

f, u = pd.factorize(new_label_cols)

y_test  = np.array(

    train[new_label_cols]

)

train["target"]=[', '.join(u[y.astype(bool)]) for y in y_test]



# train["target"]=

# labels = [', '.join(u[y.astype(int)]) for y in y_test]
train.shape
train.head()
train[label_cols].max(axis=1).describe()
len(train),len(test)
# COMMENT = 'comment_text'

# train[COMMENT].fillna("unknown", inplace=True)

# test[COMMENT].fillna("unknown", inplace=True)
df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
pd.concat([train, test], axis=0).drop_duplicates(subset='comment_text').drop("id",axis=1).to_csv('toxic_raw_text.csv.gz', index=False,compression="gzip",quoting=csv.QUOTE_ALL)
train[["id",'comment_text',"target"]].to_csv('train_toxic_raw_v0.csv.gz', index=False,compression="gzip",quoting=csv.QUOTE_ALL)
n = train.shape[0]

vec = CountVectorizer(ngram_range=(1,2),min_df=3, max_df=0.97,max_features = 60000) # could also try adding stop word removals, stemming, not lowercasing!



vec.fit(df.values)

trn_term_doc = vec.transform(train[COMMENT])

test_term_doc = vec.transform(test[COMMENT])



# trn_term_doc = vec.fit_transform(train[COMMENT])

# test_term_doc = vec.transform(test[COMMENT])
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x=trn_term_doc.sign()

test_x = test_term_doc.sign()
def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=0.1, dual=True) # ORIG

#     m = LogisticRegressionCV(Cs=12)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
preds = np.zeros((len(test), len(label_cols)))
# for i, j in enumerate(label_cols):

#     print('fit', j)

#     m,r = get_mdl(train[j])

#     preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
# submid = pd.DataFrame({'id': subm["id"]})

# submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

# submission.to_csv('submission.csv', index=False)