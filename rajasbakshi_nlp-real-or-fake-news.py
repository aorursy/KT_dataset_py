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



# Any results you write to the current directory are saved as output.
df_true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

df_fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
df_true.head()
df_true["true"] = 1

df_fake["true"] = 0



df = pd.concat([df_true,df_fake])

df.head()
df["title"] = df["title"]+" "+df["text"]+" "+df["subject"]

df.head()
del df['date']

del df['text']

del df['subject']
df.head()
import matplotlib.pyplot as plt 

from nltk.corpus import stopwords

from wordcloud import WordCloud

plt.figure(figsize = (20,20))
wc = WordCloud (max_words = 30000, stopwords = set(stopwords.words("english"))).generate(" ".join(df.title))

plt.imshow(wc)
import re

from nltk.stem.porter import PorterStemmer

X = df.title.values

y = df.true.values

ps = PorterStemmer()



corpus = []



for i in range (len(X)):

    sent = re.sub("[^A-Za-z]", " ", X[i])

    sent = sent.lower().split()

    sent = [word for word in sent if word not in set(stopwords.words('english'))]

    sent = " ".join(sent)    

    corpus.append(sent)
len(corpus)
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(max_features = 10000)

X_processed = cv.fit_transform (corpus).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size= 0.2,random_state = 10)
test_size = 0.2

X_train = X_processed[:int(len(X_processed)*(1-test_size))]

X_test = X_processed[int(len(X_processed)*(1-test_size)):]

y_train = y[:int(len(X_processed)*(1-test_size))]

y_test = y[int(len(X_processed)*(1-test_size)):]

from sklearn.model_selection import StratifiedKFold

from keras.layers import Dense

from keras.models import Sequential

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = 10)

cvscores = []



for train, test in kfold.split(X_train, y_train):

    ann = Sequential ()

    ann.add(Dense(output_dim = 128,  activation = 'relu', input_dim = 10000))

    ann.add(Dense(output_dim = 4,  activation = 'relu', input_dim = 10000))

    ann.add(Dense(units = 1 , activation = 'sigmoid'))



    ann.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    ann.fit(X_train[train], y_train[train], batch_size = 100,epochs = 5)

    

    scores = ann.evaluate(X_train[test], y_train[test], verbose=0)

    print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)