import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(train_df.head())
print("\n")
print(len(train_df))
print(test_df.head())
print("\n")
print(len(test_df))
edited_train_df = train_df.dropna(how='any', subset=['keyword'])
print(edited_train_df)
edited_test_df = test_df.dropna(how='any', subset=['keyword'])
print(edited_test_df)
train_df[train_df["target"] == 0]["text"].values[22]
train_df[train_df["target"] == 1]["text"].values[22]
tfidf_v = TfidfVectorizer()
train_model = tfidf_v.fit_transform(train_df["text"])

test_model = tfidf_v.transform(test_df["text"])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_model, train_df["target"], cv=3, scoring="f1")
scores
clf.fit(train_model, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_model)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
