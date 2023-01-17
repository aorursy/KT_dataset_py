import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
wrdf = pd.read_csv("../input/winemag-data_first150k.csv")

wrdf2 = pd.read_csv('../input/winemag-data-130k-v2.csv')



wrdf=pd.concat([wrdf,wrdf2])
# Let's drop rows with NaN price

# It's used slightly later



wr = wrdf.dropna(subset=['price'],axis=0)

wr.head(3)
wrdf.groupby([ 'country','title'])['points'].mean().sort_values(ascending=False).head(20)
wrdf.groupby(['country','winery'])['points'].mean().sort_values(ascending=False).head(10)
wrdf.groupby(['country'])['points'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=[8,5])

plt.scatter(x='points',y='price',data=wr,c='red', edgecolors='blue',marker='.')
plt.figure(figsize=[8,5])

wr['price'].value_counts().hist(bins=50)
np.corrcoef(wr['price'],wr['points'])
per = wrdf.dropna(subset=['taster_name'],axis=0)

lpper = per.groupby(['taster_name'])['points'].mean().sort_values(ascending=True).head(1)

print(lpper)
mpper = per.groupby(['taster_name'])['points'].mean().sort_values(ascending=False).head(1)

print(mpper)
plt.figure(figsize=[16,5])

plt.scatter(x=per['taster_name'], y = per['points'],marker='.')

plt.tight_layout

plt.xticks(rotation=70)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer



cvectorizer = CountVectorizer()



Vecdescr = cvectorizer.fit_transform(wrdf['description'])



X = Vecdescr
y = wrdf['points'].apply(lambda x: round(x/4)-20)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier(verbose=True,max_iter=10)
mlpc.fit(X_train,y_train)
predictions = mlpc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predictions))