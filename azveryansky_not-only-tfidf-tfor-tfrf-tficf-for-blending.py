import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from textvec import vectorizers


train = pd.read_csv('../input/train.csv').fillna(' ')#.sample(10000, random_state=13)
train_target = train['target'].values

train_text = train['question_text']

X_train, X_test, y_train, y_test = train_test_split(train_text, train_target, test_size=0.1, random_state=13)

count_vec = CountVectorizer(strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1)).fit(X_train)

tfor_vec = vectorizers.TforVectorizer(sublinear_tf=True)
tfor_vec.fit(count_vec.transform(X_train), y_train)
train_or, ci_95 = tfor_vec.transform(count_vec.transform(X_train), confidence=True)
test_or = tfor_vec.transform(count_vec.transform(X_test))

classifier = LogisticRegression(C=10, solver='sag', random_state=13)
classifier.fit(train_or, y_train)
val_preds = classifier.predict_proba(test_or)[:,1]
print('ROC_AUC -> ', roc_auc_score(y_test, val_preds))
print('shape -> ', train_or.shape)
classifier = LogisticRegression(C=10, solver='sag', random_state=13)
classifier.fit(train_or[:,ci_95], y_train)
val_preds = classifier.predict_proba(test_or[:,ci_95])[:,1]
print('ROC_AUC -> ', roc_auc_score(y_test, val_preds))
print('shape -> ', train_or[:,ci_95].shape)
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from textvec import vectorizers


train = pd.read_csv('../input/train.csv').fillna(' ')#.sample(100000, random_state=13)
test = pd.read_csv('../input/test.csv').fillna(' ')#.sample(10000, random_state=13)
test_qid = test['qid']
train_target = train['target'].values

train_text = train['question_text']
test_text = test['question_text']

tfidf_vec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1))
tfidf_vec.fit(pd.concat([train_text, test_text]))
train_idf = tfidf_vec.transform(train_text)


count_vec = CountVectorizer(strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1)).fit(train_text)

tfrf_vec = vectorizers.TfrfVectorizer(sublinear_tf=True)
tfrf_vec.fit(count_vec.transform(train_text), train_target)
train_rf = tfrf_vec.transform(count_vec.transform(train_text))

tfor_vec = vectorizers.TforVectorizer(sublinear_tf=True)
tfor_vec.fit(count_vec.transform(train_text), train_target)
train_or = tfor_vec.transform(count_vec.transform(train_text))

tficf_vec = vectorizers.TfIcfVectorizer(sublinear_tf=True)
tficf_vec.fit(count_vec.transform(train_text), train_target)
train_icf = tficf_vec.transform(count_vec.transform(train_text))

tfbinicf_vec = vectorizers.TfBinIcfVectorizer(sublinear_tf=True)
tfbinicf_vec.fit(count_vec.transform(train_text), train_target)
train_binicf = tfbinicf_vec.transform(count_vec.transform(train_text))

results = {}

def validate_results(train_data_vecs, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    for i, (train_index, val_index) in enumerate(skf.split(train_text, train_target)):
        x_train, x_val = train_data_vecs[list(train_index)], train_data_vecs[list(val_index)]
        y_train, y_val = train_target[train_index], train_target[val_index]
        classifier = LogisticRegression(C=10, solver='sag', random_state=13)
        classifier.fit(x_train, y_train)
        val_preds = classifier.predict_proba(x_val)[:,1]
        current_results = results.get(name,{'preds': [], 'target': []})
        current_results['preds'].extend(val_preds)
        current_results['target'].extend(y_val)
        results[name] = current_results

validate_results(train_rf, 'rf')
validate_results(train_idf, 'idf')
validate_results(train_or, 'or')
validate_results(train_binicf, 'binicf')
validate_results(train_icf, 'icf')
import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline
res = []
for k, v in results.items():
    res.append((k, roc_auc_score(v['target'],np.array(v['preds'])) ,v['preds']))
res = sorted(res, key= lambda x:-x[1])
corrs = np.corrcoef(list(zip(*res))[2])
accs = list(zip(*res))[1]
labels = [f'{x}:{accs[i]:.4f}' for i, x in enumerate(list(zip(*res))[0])]
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.heatmap(corrs, 
                 linewidth=0.5, 
                 annot=corrs, 
                 square=True, 
                 ax=ax, 
                 xticklabels=labels,
                 yticklabels=labels)

plt.show()
