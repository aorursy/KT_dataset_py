import pandas as pd

import gc
Train = pd.read_csv('../input/texts-classification-iad-hse-intro-2020/train.csv')
Test = pd.read_csv('../input/texts-classification-iad-hse-intro-2020/test.csv')
Train.head()
Test.head()
Train.isnull().sum()
Test.isnull().sum()
Train.fillna('', inplace=True)
Test.fillna('', inplace=True)
Train['title&description'] = Train['title'].str[:] + ' ' + Train['description'].str[:]
Test['title&description'] = Test['title'].str[:] + ' ' + Test['description'].str[:]
Train.drop(columns=['title', 'description'], inplace=True)

Test.drop(columns=['title', 'description'], inplace=True)
gc.collect()
import nltk



nltk.download('stopwords')



from nltk.corpus import stopwords



ru_stopwords = list(stopwords.words("russian"))
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(stop_words=ru_stopwords)

tf_idf.fit(Train['title&description'])
from sklearn.svm import LinearSVC
Train_tf_idf = tf_idf.transform(Train['title&description'])

Test_tf_idf = tf_idf.transform(Test['title&description'])
clf = LinearSVC()

clf.fit(Train_tf_idf, Train['Category'])
del Train, Train_tf_idf

gc.collect()
Answer = pd.DataFrame(columns=['Id', 'Category'])

Answer['Id'] = Test['itemid']
Answer['Category'] = clf.predict(Test_tf_idf)
Answer.to_csv('my_submission_tfidfvect_nltk_stopwords_lsvc.csv', index=None)