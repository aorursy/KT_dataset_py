import os
import numpy as np 
import pandas as pd 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import CountVectorizer


path = "/kaggle/input/nlp-getting-started/"
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_test = pd.read_csv(os.path.join(path, 'test.csv'))
df_sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
df_train_true = df_train[df_train.target == 1]
df_train_false = df_train[df_train.target == 0]


df_train_true.text.apply(lambda x: len(x)).plot.hist(alpha=0.4, color="blue")
df_train_false.text.apply(lambda x: len(x)).plot.hist(alpha=0.4, color="red")


df_train_true.text.apply(lambda x: len(x.split())).plot.hist(alpha=0.4, color="blue")
df_train_false.text.apply(lambda x: len(x.split())).plot.hist(alpha=0.4, color="red")

df_train_false.location.value_counts()
df_train_false.keyword.value_counts()

cv = CountVectorizer()
cv.fit(df_train_true.text)
cv.fit(df_train_false.text)


cv_fit_t = cv.transform(df_train_true.text)
top_words = {name: occurence for name, occurence in zip(cv.get_feature_names(), cv_fit_t.toarray().sum(axis=0))}
cv_fit_f = cv.transform(df_train_false.text)



data = {'feature': cv.get_feature_names(), 
        'true': cv_fit_t.toarray().sum(axis=0), 
        "false":  cv_fit_f.toarray().sum(axis=0)}

df_words = pd.DataFrame(data=data)
df_words.set_index(['feature'])
df_words.sort_values(by="false", ascending=True).head(100)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.4, min_df=2, stop_words = 'english')),
    ('clf', SVC(random_state=0, tol=1e-3))
])
X_train, X_test, y_train, y_test = train_test_split(df_train.text, df_train.target, test_size=0.2, random_state=1)

pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
print(metrics.accuracy_score(y_test, prediction))
print(metrics.precision_score(y_test, prediction))




df_sub["target"] = pipeline.predict(df_test.text)
df_sub.to_csv("submission.csv",index=False)
