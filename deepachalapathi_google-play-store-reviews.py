# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
data.head()
dataset=pd.concat([data.Translated_Review,data.Sentiment],axis=1)
dataset.dropna(axis=0,inplace=True)
dataset.head()
corpus=[]
for i in dataset.Translated_Review:
    text=re.sub("[^a-zA-Z]"," ",i)
    text=text.lower()
    text=nltk.word_tokenize(text)
    lemma=WordNetLemmatizer()
    text=[lemma.lemmatize(word) for word in text]
    text=" ".join(text)
    corpus.append(text)
#Printing the unique values in the 'Sentiment' column
print(np.unique(dataset['Sentiment']))
#encoding the 'Sentiment' column
dataset.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in dataset.Sentiment]

#creating of a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary = True, stop_words = 'english', max_features = 200000)
sparce_matrix = cv.fit_transform(corpus).toarray()
all_words=cv.get_feature_names()
#creating wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.subplots(figsize=(10,10))
wordcloud=WordCloud(background_color="black",width=1024,height=768).generate(" ".join(all_words[100:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#splitting the dataset into test and train sets
X = sparce_matrix
y = dataset.iloc[:,1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rclf = RandomForestClassifier(n_estimators = 10, random_state=42)
rclf.fit(X_train,y_train)
print("Accuracy %: ",rclf.score(X_test,y_test)*100)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
print("Accuracy %: ",lr.score(X_test,y_test)*100)