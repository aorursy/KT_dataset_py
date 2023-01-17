import numpy as np

import pandas as pd
data = pd.read_csv("/kaggle/input/youtubevideodataset/Youtube Video Dataset.csv")

data
data = data.drop(["Videourl","Description"],axis=1)

data
data is None
data.info()
data.Category.value_counts()
data["Category"] = data["Category"].map({"travel blog":0,"Science&Technology":1,"Food":2,"Art&Music":3,"manufacturing":4,"History":5})

data
import pandas as pd

import numpy as np

import nltk 

import re

from nltk.corpus import stopwords



title_list = []

for title in data.Title:

    title = re.sub("[^a-zA-Z]"," ", title)

    title = title.lower()

    title = nltk.word_tokenize(title)

    lemma = nltk.WordNetLemmatizer()

    title = [ lemma.lemmatize(word) for word in title]

    title = " ".join(title)

    title_list.append(title)
from sklearn.feature_extraction.text import CountVectorizer

max_features = 1000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")

space_matrix = count_vectorizer.fit_transform(title_list).toarray() # 0-1
y = data["Category"].values

y
x = space_matrix

x
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

print("x_train",x_train.shape)

print("x_test",x_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(x_train,y_train)



print("Accuracy => ", nb.score(x_test,y_test)*100)
all_words = count_vectorizer.get_feature_names()

print("Most used words: ",all_words[50:100])
from wordcloud import WordCloud

import matplotlib.pyplot as plt

plt.subplots(figsize=(12,12))

wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(all_words[100:]))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10, random_state=42)

rf.fit(x_train,y_train)

print("accuracy: ",rf.score(x_test,y_test)*100)
#confussion matrix

y_pred=rf.predict(x_test)

y_true=y_test



from sklearn.metrics import confusion_matrix

import seaborn as sns

names=["travel blog","Science&Technology","Food","Art&Music","manufacturing","History"]

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

ax.set_xticklabels(names,rotation=90)

ax.set_yticklabels(names,rotation=0)

plt.show()