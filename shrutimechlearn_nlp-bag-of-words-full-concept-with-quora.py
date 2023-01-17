import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df_train = pd.read_csv("../input/train.csv",index_col='qid')

df_test = pd.read_csv("../input/test.csv",index_col='qid')

#df = pd.concat([df_train ,df_test],sort=True)
df_train.info()
df_train.head()
df_train.target.value_counts()
p = sns.countplot(x=None,y= 'target',data=df_train)
df_train.dropna(inplace=True)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vect = CountVectorizer()

vect
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgbm





text_clf_lgbm = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', lgbm.LGBMClassifier()),

])

from sklearn.model_selection import train_test_split



X = df_train['question_text']

y = df_train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)
text_clf_lgbm.fit(X_train, y_train)

# Form a prediction set

predictions = text_clf_lgbm.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, predictions)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
df_test['prediction'] = text_clf_lgbm.predict(df_test['question_text'])

df_test.head()
df_test.drop(['question_text'],axis=1,inplace=True)

df_test.to_csv('submission.csv',index=True)
df_test.head()