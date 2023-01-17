import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
def save_csv(qid,label):
    with open('submission.csv','a') as f:
        f.write(qid+','+label)
        f.write("\n")
    f.close()
    
with open('submission.csv','a') as f:
    f.write('qid,prediction')
    f.write("\n")
f.close()
# Train data
df = pd.read_csv('../input/train.csv')
df.isnull().any()
X_train = df.question_text
y_train = df.target.astype(str)
lb = preprocessing.MultiLabelBinarizer()
y_train = lb.fit_transform(y_train)

# Test data
df_test = pd.read_csv('../input/test.csv')
classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_df=0.5, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, y_train)
for f, b in zip(df_test.qid, df_test.question_text):
    X_test = np.array([b])
    predicted = classifier.predict(X_test)    
    count = 0
    label = None
    
    for i in predicted[0]:
        if i == 1:
            label = count
        count += 1

    save_csv(str(f),str(label))