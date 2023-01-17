import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df = df.rename(columns = {'v1':'Class','v2':'Text'})

df.head()
df["Class"]=df["Class"].apply(lambda x:1 if x=="spam" else 0)
df.head()
sns.countplot(x="Class",data=df)
df["length"]=df["Text"].apply(lambda x: len(x))
df.head()
sns.kdeplot(df[df["Class"]==0]["length"],shade=True,label="not spam")

plt.xlabel("length")

sns.kdeplot(df[df["Class"]==1]["length"],shade=True,label="spam")

from sklearn.feature_extraction.text import CountVectorizer

df[df["length"]>500]["Text"].iloc[0]
cv=CountVectorizer().fit_transform(df["Text"])
#print(cv)
y=df["Class"]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(cv,y,test_size=0.3)
from sklearn.naive_bayes import MultinomialNB
algo=MultinomialNB()

algo.fit(x_train,y_train)
pre=algo.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pre))