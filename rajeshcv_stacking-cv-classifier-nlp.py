import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier



from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

import lightgbm as lgb

import xgboost as xgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score,f1_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
tfid_transformer= feature_extraction.text.TfidfTransformer()

train_tfid = tfid_transformer.fit_transform(train_vectors)

test_tfid = tfid_transformer.fit_transform(test_vectors)
# clf1 = ExtraTreesClassifier(n_estimators=500,criterion='entropy',random_state=42,max_features=None,min_samples_leaf=3)

# clf2 = GradientBoostingClassifier(n_estimators=500,learning_rate=.08,max_depth=10,random_state=42)

# clf3 =  OneVsRestClassifier(XGBClassifier(n_estimators = 719,

#                                                    max_depth = 464,

#                                                    random_state=42))

# clf4 = lgb.LGBMClassifier(n_estimators=500,random_state=42,learning_rate=.08,max_depth=10)
clf1 = linear_model.RidgeClassifier(alpha=3,class_weight='balanced',random_state=42,solver='saga',fit_intercept=False)

clf2 = linear_model.PassiveAggressiveClassifier(random_state=42)

clf3 = linear_model.LogisticRegression(random_state=42,penalty='elasticnet',solver='saga',l1_ratio=0.5,n_jobs=-1,

                                       max_iter=1000,class_weight='balanced')

clf4 = linear_model.SGDClassifier(loss='log',penalty='elasticnet',random_state=42)
scores = model_selection.cross_val_score(clf1, train_tfid, train_df["target"], cv=3, scoring="f1",n_jobs=-1)

scores
scores = model_selection.cross_val_score(clf2, train_tfid, train_df["target"], cv=3, scoring="f1",n_jobs=-1)

scores
scores = model_selection.cross_val_score(clf3, train_tfid, train_df["target"], cv=3, scoring="f1",n_jobs=-1)

scores
scores = model_selection.cross_val_score(clf4, train_tfid, train_df["target"], cv=3, scoring="f1",n_jobs=-1)

scores
stackmodel= StackingCVClassifier(classifiers=[clf1,clf3,clf4],

                             meta_classifier=clf1,

                             cv=3,

                             use_probas=False, 

                             use_features_in_secondary=False,

                             verbose=-2,

                             n_jobs=-1)
scores = model_selection.cross_val_score(stackmodel, train_tfid, train_df["target"], cv=3, scoring="f1",n_jobs=-1)

scores
stackmodel.fit(train_tfid, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = stackmodel.predict(test_tfid)
len(sample_submission[sample_submission.target==1])
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)