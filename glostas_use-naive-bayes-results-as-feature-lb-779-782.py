# code snippet
"""
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgbm

# Perform naive Bayes classification
nb = GaussianNB()
nb.fit(features,labels)
nb_res  = gau.predict_proba(features)[:,1]

# add results of Bayes classification to feature
features  = np.c_[features,nb_res]
test_feat = np.c_[test_feat,nb.predict_proba(test_feat)]

# Perform LGBM classification
clf = lgbm.LGBMClassifier(**lgbm_params)
clf.fit(features,labels)
results = clf.preict_proba(test_feat)
"""
