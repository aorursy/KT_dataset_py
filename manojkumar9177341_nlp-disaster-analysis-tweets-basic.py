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
import re
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train.head()
target = df_train['target']
df_train.drop('target', axis=1, inplace=True)
join_df = [df_train, df_test]

both_df = pd.concat(join_df, sort=True, keys=['x','y'])
both_df.head()
no_na_df = pd.DataFrame()
no_na_df['text'] =  both_df['text'] + both_df['keyword'].apply(lambda x: ' '+str(x)) + both_df['location'].apply(lambda x: ' '+str(x))
no_na_df['text'] = no_na_df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
no_na_df
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stop = stopwords.words('english')
from nltk.stem import PorterStemmer

pst = PorterStemmer()
no_na_df['text'] = no_na_df['text'].apply(lambda x: " ".join(re.split("[^a-zA-Z]*", x)) if x else '')
no_na_df['text'] = no_na_df['text'].apply(lambda x: x.lower())
no_na_df
no_na_df['text'] = no_na_df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x.split() if i not in stop]))
no_na_df['text'] = no_na_df['text'].apply(lambda x: ' '.join([pst.stem(i) for i in x.split() if i not in stop]))
corpus = np.array(no_na_df['text'])
corpus1 = np.array(no_na_df.loc['y']['text'])
len(no_na_df.loc['x']['text'])
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1000000)

X = cv.fit_transform(corpus).toarray()

y = cv.transform(corpus1).toarray()
len(X)
df_submit = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

df_submit.head()
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X[:7613], target.values, test_size=0.8, random_state=0)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
gnb = RandomForestClassifier(max_depth=10, random_state=0)

classifier = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))
gnb = DecisionTreeClassifier(random_state=0)

classifier = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))
gnb = RidgeClassifier()

classifier = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))
gnb = GaussianNB()

classifier = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))
gnb = MultinomialNB()

classifier = gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(sum([1 for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]),len(y_pred))
gnb = MultinomialNB()

classifier = gnb.fit(X[:7613], target.values)

y_pred = gnb.predict(X[7613:])
df_submit['target'] = y_pred
df_submit.to_csv('sample_submission.csv', index=False)