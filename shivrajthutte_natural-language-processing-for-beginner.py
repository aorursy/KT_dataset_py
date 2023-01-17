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
df=pd.read_csv("/kaggle/input/deepnlp/Sheet_1.csv")

df.head()
df_data=df.drop(["Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7"],axis="columns")

df_data.count()
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt

text_df = " ".join(review for review in df_data.response_text)

wordcloud = WordCloud(background_color="red").generate(text_df)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer

vectorizer = CountVectorizer(stop_words='english')

docs       = vectorizer.fit_transform(text for text in df_data['response_text'])

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(

    features=features, size=(1080, 720)

)

visualizer.fit(docs)

visualizer.show()
import re

from nltk.corpus import stopwords

stops = stopwords.words('english')
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 80):

    response_text = re.sub('[^a-zA-Z]', ' ', df_data['response_text'][i])

    response_text = response_text.lower()

    response_text = response_text.split()

    ps = PorterStemmer()

    response_text = [ps.stem(word) for word in response_text if not word in set(stops)]

    response_text = ' '.join(response_text)

    corpus.append(response_text)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1000)

X = cv.fit_transform(corpus).toarray()
y_response = df_data.iloc[:, 1]
y=np.where(y_response=="flagged",0,1)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4, 

                                                    random_state=1,

                                                    stratify=y)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=1211,

                                  n_estimators=500,oob_score=True)
model_rf.fit( X_train , y_train )

y_pred_probarf = model_rf.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_probarf)