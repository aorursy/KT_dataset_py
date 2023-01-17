import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import RidgeClassifier,LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

import re
df=pd.read_csv("../input/nlp-getting-started/train.csv")

df_test=pd.read_csv("../input/nlp-getting-started/test.csv")

df1=df.copy()

df1.head()
print(df1.shape, df_test.shape)
print('-'*50)

print("Training:\n")

print(df1.info())

print('-'*50)

print("Testing:\n")

print(df_test.info())
df1.drop(columns=['id'],inplace=True)

df1.head()
df1['len']=df1['text'].str.len()

df1.head()
sns.barplot(x='target',y='len',data=df1)

plt.show()
for df in range(7613):

    df1['hashtag']=df1.text.str.findall(r'#\w+')

df1['hash_count']=df1.hashtag.str.len()

df1.head()
sns.barplot(x='hash_count',y='target',data=df1)

plt.title("Is hash count affect target")

plt.show()
sns.countplot(x='target',data=df1).set_title("No. of Disaster tweets")

plt.show()
df1.describe()
count=CountVectorizer()

tf=TfidfVectorizer()

train_vectors=count.fit_transform(df1['text'][0:5])

train_vectors[0].todense()
train_vectors=count.fit_transform(df1["text"])

test_vectors=count.transform(df_test["text"])



clf=RidgeClassifier()

scores=cross_val_score(clf,train_vectors,df1['target'],cv=3,scoring='f1')

scores.mean()
clf.fit(train_vectors,df1['target'])

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv") 

submission['target']=clf.predict(test_vectors)

submission.head()
submission.to_csv("submission.csv",index=False)