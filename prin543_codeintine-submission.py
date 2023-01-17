# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
df.head()
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1, inplace=True)
df.head()
ax=df['v1'].value_counts().plot.bar()
ax.set(xlabel="Message Categories", ylabel="Count", title="Value Counts for each category")
import re
    
##preprocess the tweets
def preprocess_text(t):
    t=str(t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"\_", " ", t)
    t = t.lower()
    t = re.sub(r"\W"," ",t)
    t = re.sub(r"\s+"," ",t)
    t = re.sub(r"\s+[a-z]\s+"," ",t)
    t = re.sub(r"\s+[a-z]$"," ",t)
    t = re.sub(r"^[a-z]\s+"," ",t)
    t = re.sub(r"\d"," ",t)
    t = re.sub(r"\s+"," ",t)

    return t

df.v2=df.v2.apply(preprocess_text)
df.head()
X=df.v2
y=df.v1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100,learning_rate=0.1)


my_clf=Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features = 500, min_df = 1, max_df = 0.8, stop_words = stopwords.words('english'),
                              preprocessor=preprocess_text)),
    ('model',model)
])
my_clf.fit(X_train,y_train)
preds=my_clf.predict(X_test)
preds
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, preds)
from sklearn.metrics import classification_report
target_names = ['Ham', 'Spam']
print(classification_report(y_test, preds, target_names=target_names))
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, preds)