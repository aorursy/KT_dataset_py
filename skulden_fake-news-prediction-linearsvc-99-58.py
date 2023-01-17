import numpy as np

import pandas as pd 

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

true_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')



fake_df['target'] = 'fake'

true_df['target'] = 'true'



news_df = pd.concat([fake_df, true_df]).reset_index(drop = True)



news_df.head()
x_train,x_test,y_train,y_test = train_test_split(news_df['text'], news_df.target, test_size=0.1, random_state=212)



pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LinearSVC())])



model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)



print("accuracy score: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))