import numpy as np

import pandas as pd 

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv', encoding = "ISO-8859-1", names = ['target', 'text'])



df.head()
x_train,x_test,y_train,y_test = train_test_split(df['text'], df.target, test_size=0.1, random_state = 212)



pipe = Pipeline([('tfidf', TfidfVectorizer()),

                 ('model', LinearSVC())])



model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)



print("accuracy score: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))