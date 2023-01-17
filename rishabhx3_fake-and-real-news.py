import numpy as np

import pandas as pd



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from string import punctuation



from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
real.head()
fake.head()
real['category']=1

fake['category']=0
df = pd.concat([real,fake])
df.isna().sum()
df['title'].count()
df.subject.value_counts()
df['text'] = df['text'] + " " + df['title'] + " " + df['subject']

del df['title']

del df['subject']

del df['date']
stop = set(stopwords.words('english'))

pnc = list(punctuation)

stop.update(pnc)
stemmer = PorterStemmer()

def stem_text(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            word = stemmer.stem(i.strip())

            final_text.append(word)

    return " ".join(final_text)
df['text'] = df['text'].apply(stem_text)
X_train,X_test,y_train,y_test = train_test_split(df['text'],df['category'])
cv = CountVectorizer(min_df=0,max_df=1,ngram_range=(1,2))



cv_train = cv.fit_transform(X_train)

cv_test = cv.transform(X_test)



print('Train shape: ',cv_train.shape)

print('Test shape: ',cv_test.shape)
nb = MultinomialNB()
nb.fit(cv_train, y_train)
pred_nb = nb.predict(cv_test)
score = metrics.accuracy_score(y_test, pred_nb)

print("Accuracy Score: ",score)