%%time

# This is my first time working on Kaggle. 

# I have been following the 'Kernels' everyone has been sharing, while getting up to speed on my own

# Thank you to jazivxt, paulorzp, cdeotte for solutions dueling in elegance 

# and Allunia, Abhishek, speedwagon and many others for exploring and sharing early

# I plan to make original contributions soon

#

# This submission is just an exercise to check the submission process, so the only change in this fork

# is the substitution of Gradient Boosted Regressior in place of SVM



import pandas as p; from sklearn import *

import warnings; warnings.filterwarnings("ignore")

t, r = [p.read_csv('../input/' + f) for f in ['train.csv', 'test.csv']]

cl = 'wheezy-copper-turtle-magic'; re_ = []

col = [c for c in t.columns if c not in ['id', 'target', cl]]



# These hyperparameters have not been tested. They are mostly default, but configured to take advantage of early stopping

gbc = ensemble.GradientBoostingClassifier(learning_rate=0.1,

                                  n_estimators=500,

                                  subsample=1.0,

                                  max_depth=3,

                                  presort=True,

                                  validation_fraction=0.1,

                                  n_iter_no_change=7,

                                  random_state=7)

for s in sorted(t[cl].unique()):

    t_ = t[t[cl]==s]

    r_ = r[r[cl]==s]

    gbc.fit(t_[col], t_['target'])

    r_['target'] = gbc.predict_proba(r_[col])[:,1]

    re_.append(r_)

p.concat(re_)[['id','target']].to_csv("submission.csv", index=False)