import pandas as pd
import numpy as np
import re
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/train.csv')
df.head()
print(df['target'].value_counts(), end = '\n\n')
print(sum(df['target'] == 1) / sum(df['target'] == 0) * 100, 'percent of questions are insincere.')
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

train_clean = preprocess_reviews(train['question_text'])
test_clean = preprocess_reviews(test['question_text'])
cv = CountVectorizer(binary=True)
cv.fit(train_clean)
X_train = cv.transform(train_clean)
X_test = cv.transform(test_clean)
target_train = train['target']
target_test = test['target']

model = LogisticRegression()
model.fit(X_train, target_train)
print("Accuracy: %s" % accuracy_score(target_test, model.predict(X_test)))
kaggle_test = pd.read_csv('../input/test.csv')
kaggle_test_clean = preprocess_reviews(kaggle_test['question_text'])
X_kaggle_test = cv.transform(kaggle_test_clean)
results = model.predict(X_kaggle_test)
submission = pd.DataFrame({"qid" : kaggle_test['qid'], "prediction" : results})
submission.to_csv("submission.csv", index=False)