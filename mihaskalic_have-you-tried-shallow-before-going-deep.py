import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
train_df = pd.read_csv("../input/train.csv", nrows=100000)
train_txt = train_df["question_text"]

test_df = pd.read_csv("../input/test.csv")
test_txt = test_df["question_text"]
# Feature extraction
tfidf = TfidfVectorizer(
    min_df=5, max_features=10000, strip_accents='unicode',lowercase =True,
    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, stop_words = 'english'
).fit(train_txt)

X_tr = tfidf.transform(train_txt)
X_te = tfidf.transform(test_txt)
y = train_df["target"]
# Classification and prediction
clf = LogisticRegression(C=3)
clf.fit(X_tr, y)

p_test = clf.predict_proba(X_te)[:, 0]
y_te = (p_test > 0.5).astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)