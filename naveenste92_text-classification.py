import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
df=pd.read_csv("../input/Sheet_1.csv")
df.head()
df=df.drop(df.columns[[0,3,4,5,6,7]],axis=1)
df.columns=['value','response']
df['category_id']=df['value'].factorize()[0]
df.head()

df.groupby('value').response.count().plot.bar()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df['response'], df['value'], random_state = 0)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LinearSVC().fit(X_train_tfidf, y_train)
print(clf.predict(count_vect.transform(["On the memorial anniversary of my friends father I was with him to give support and remind him that he's not alone."])))
df[df['response'] =="On the memorial anniversary of my friends father I was with him to give support and remind him that he's not alone."] 
print(clf.predict(count_vect.transform(["dont know"])))