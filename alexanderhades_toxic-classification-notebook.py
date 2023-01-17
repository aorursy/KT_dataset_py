# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score

%matplotlib inline
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/train.csv.zip")
df.head(3)
for i,col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    plt.title(col)
    plt.bar([col,"Not "+col], np.sort(df[col].value_counts()))
    plt.show()
X_train, X_test, y_train, y_test = train_test_split(df["comment_text"], df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size=0.33, random_state=42)
cv = CountVectorizer()
train_input = cv.fit_transform(X_train)
test_input = cv.transform(X_test)
print("Input BOW shape ", train_input.shape)
'''
LighGBM req. float32/64 (std. int64)
'''
train_input = train_input.astype("float")
test_input = test_input.astype("float")
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

y = y_train.toxic

mnb = MultinomialNB()

parameter_grid = {}

scoring = {'AUC': 'roc_auc', 
           'f1': 'f1',
           'Accuracy': make_scorer(accuracy_score)}

clf = GridSearchCV(mnb,
                   parameter_grid,
                   cv=3,
                   scoring=scoring,
                   refit='AUC', 
                   return_train_score=True)

clf.fit(train_input, y)

print("Accuracy test score: \t", clf.cv_results_['mean_test_Accuracy'][0])
print("AUC test score: \t", clf.cv_results_['mean_test_AUC'][0])
print("F1 test score: \t", clf.cv_results_['mean_test_f1'][0])

y_pred = clf.predict(test_input)
print(confusion_matrix(y_pred, y_test.toxic))
'''
Pipeline for the entuire pipeline (text -> Classification)
For the purpose of testing custom input on demand 
'''

y = y_train.toxic

clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("MNB", MultinomialNB())
])

parameter_grid = {}

scoring = {'AUC': 'roc_auc', 
           'f1': 'f1',
           'Accuracy': make_scorer(accuracy_score)}

clf = GridSearchCV(clf,
                   parameter_grid,
                   cv=3,
                   scoring=scoring,
                   refit='AUC', 
                   return_train_score=True)

clf.fit(X_train, y)
print("Accuracy test score: \t", clf.cv_results_['mean_test_Accuracy'][0])
print("AUC test score: \t", clf.cv_results_['mean_test_AUC'][0])
print("F1 test score: \t", clf.cv_results_['mean_test_f1'][0])

y_pred = clf.predict(X_test)
print(confusion_matrix(y_pred, y_test.toxic))
'''
test above pipelien below
'''

print(clf.predict(["you are an idiot"]))
print(clf.predict(["I am sick"]))
print(clf.predict(["youre a moron"]))
'''
Naive classifiers per label
'''

res_clf = {}
for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    print(col)
    y = y_train[col]
    parameterGrid = {}
    mnb = MultinomialNB()
    clf = GridSearchCV(mnb, parameterGrid, cv=10)
    clf.fit(train_input, y)
    print("Mean test score: \t", clf.cv_results_["mean_test_score"][0])
    res_clf[col] = clf
for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    print(col)
    y_true = y_test[col]
    y_pred = res_clf[col].predict(test_input)
    print("Accuracy: ", f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
pd.read_csv("../input/sample_submission.csv").head()
'''
Competition submission [1/2]
'''

test_raw = pd.read_csv("../input/test.csv")
test_bow = cv.transform(test_raw["comment_text"])
res = pd.DataFrame()
res["id"]=test_raw.id
for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    y_pred = [t[1] for t in res_clf[col].predict_proba(test_bow)]
    res[col] = y_pred
res.head()
'''
Competition submission [2/2]
'''

res.to_csv('submission.csv', index=False)