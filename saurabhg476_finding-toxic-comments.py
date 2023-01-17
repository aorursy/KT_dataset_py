import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
output_columns = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
X = train['comment_text']
vectorizer = CountVectorizer(stop_words="english")
vectorizer.fit(X)
X = vectorizer.transform(X)

model = {}

for class_ in output_columns:
    y = train[class_]
    [X_train,X_test,y_train,y_test] = train_test_split(X,y,test_size=0.2)
    model[class_] = LogisticRegression(fit_intercept=False,max_iter=200)
    model[class_].fit(X_train,y_train)
    p_train = model[class_].predict(X_train)
    p_test = model[class_].predict(X_test)
    print("train f1-score for class:",class_, f1_score(y_train,p_train)) 
    print("test f1-score for class:",class_, f1_score(y_test,p_test))
    
    
    
submission = test[['id']].copy()
test_text = vectorizer.transform(test['comment_text'])
for class_ in output_columns:
    submission[class_] = model[class_].predict_proba(test_text)[:,1]    
submission.to_csv('submission.csv', index=False)
