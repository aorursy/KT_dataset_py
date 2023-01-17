import numpy as np

import pandas as pd

import re

from sklearn.svm import LinearSVC

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_extraction.text import TfidfVectorizer
cols = ['ID','title','description','timestamp','viewCount','likeCount','dislikeCount','commentCount','user_comment_1',

        'user_comment_2','user_comment_3','user_comment_4','user_comment_5','user_comment_6','user_comment_7',

        'user_comment_8','user_comment_9','user_comment_10','URL','class']
train_data = pd.read_csv('train.csv',names=cols,header=0)

train_data = train_data[['title','description','class']].copy()

test_data = pd.read_csv('test_2.csv',header=0)
train_data.head()
titleDesc = pd.DataFrame()
titleDesc['text'] = train_data[['title','description']].apply(lambda x: ' '.join(x),axis=1)
titleDesc['class'] = train_data["class"]
titleDesc.head()
stemmer = SnowballStemmer('english')

words = stopwords.words('english')
titleDesc["cleaned"] = titleDesc["text"].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]).lower())
titleDesc.head()
X_train, y_train = titleDesc["cleaned"], titleDesc["class"]

# X_train, X_test, y_train, y_test = train_test_split(titleDesc['cleaned'],train_data['outcome'],test_size=0.3)
pipeline = Pipeline([('vect',TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True)),

                    ('chi',SelectKBest(chi2,k=10000)),

                    ('clf',LinearSVC(C=1.0,penalty='l1',max_iter=3000,dual=False))])
model = pipeline.fit(X_train, y_train)
test_data['text'] = test_data[['title','description']].apply(lambda x: ' '.join(x),axis=1)
test_data["cleaned"] = test_data["text"].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]).lower())
test_data.head()
X_test = test_data["cleaned"]

Y_pred = pipeline.predict(X_test)
test_data["class"] = Y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]
result.head()
result.to_csv("test2_predictions_IanAndJordan.csv", index=False)