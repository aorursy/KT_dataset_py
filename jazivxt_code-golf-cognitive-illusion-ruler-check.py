import pandas as p; from sklearn import *

import warnings; warnings.filterwarnings("ignore")

t, r = [p.read_csv('../input/' + f) for f in ['train.csv', 'test.csv']]

cl = 'wheezy-copper-turtle-magic'; re_ = []

col = [c for c in t.columns if c not in ['id', 'target', cl]]

sv = svm.NuSVC(kernel='poly', degree=4, random_state=4, probability=True, coef0=0.08)

for s in sorted(t[cl].unique()):

    t_ = t[t[cl]==s]

    r_ = r[r[cl]==s]

    sv.fit(t_[col], t_['target'])

    r_['target'] = sv.predict_proba(r_[col])[:,1]

    re_.append(r_)

p.concat(re_)[['id','target']].to_csv("submission.csv", index=False)