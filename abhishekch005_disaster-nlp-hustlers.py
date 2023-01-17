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
import pandas as pd

import numpy as np

import plotly.graph_objects as go

from pandas_profiling import ProfileReport

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot
data = pd.read_csv('../input/nlp-getting-started/train.csv')

data_test = pd.read_csv('../input/nlp-getting-started/test.csv')
profile = ProfileReport(data, title="Pandas Profiling Report")
profile
data.target.value_counts()
#Not much class imbalance
null_columns=data.columns[data.isnull().any()]

null_ = data[null_columns].isnull().sum()

train_w_null_cols = (null_/data.shape[0])*100 

print (train_w_null_cols)
data['length'] = data['text'].apply(len)
data_1 = [

    go.Box(

        y=data[data['target']==0]['length'],

        name='Fake'

    ),

    go.Box(

        y=data[data['target']==1]['length'],

        name='Real'

    )

]

layout = go.Layout(

    title = 'Comparison of text length in Tweets '

)

fig = go.Figure(data=data_1, layout=layout)

fig.show()
data.keyword.value_counts()[:20].plot(kind='bar', title='Top 20 keywords in text', color='red')
data.location.value_counts()[:20].plot(kind='bar', title='Top 20 location in tweet', color='blue')  # Check the top 15 locations 
import re
import nltk

nltk.download('all')
from nltk.stem import WordNetLemmatizer 

ps = WordNetLemmatizer() 

from nltk.corpus import stopwords
corpus = []

for i in  range(0,7613):

    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])

    review = review.lower()

    review = review.split()

    review = [ps.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

    
corpus_test = []

for i in  range(0,3263):

    #review = re.sub('[^a-zA-Z]', ' ', data_test['text'][i])

    #review = review.lower()

    #review = review.split()

    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #review = ' '.join(review)

    corpus_test.append(review)

    
y_data = data.iloc[:,4]
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split



vect = CountVectorizer()

text_clf = Pipeline([

 ('vect', CountVectorizer()),

 ('tfidf', TfidfTransformer()),

 ('clf', SGDClassifier(loss='hinge', penalty='l2',

                       alpha=1e-3, random_state=42,

                           max_iter=5, tol=None)),])



x_train, x_test, y_train, y_test = train_test_split(data.text, y_data, test_size=0.3, random_state=0)
text_clf.fit(x_train, y_train)

predicted = text_clf.predict(x_test)

np.mean(predicted == y_test)
from sklearn.model_selection import GridSearchCV
parameters = {

     'vect__ngram_range': [(1, 1), (1, 2)],

     'tfidf__use_idf': (True, False),

     'clf__alpha': (1e-2, 1e-3),

 }
gs_clf = GridSearchCV(text_clf, parameters, cv=5)
gs_clf = gs_clf.fit(x_train, y_train)


y_pred = gs_clf.predict(x_test)
gs_clf.best_score_
from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_pred, y_test))
print(metrics.f1_score(y_test, y_pred))
gs_clf.cv_results_
data_test.head()
y_pred_test = gs_clf.predict(data_test.text)
output = pd.DataFrame({'Id': data_test.id, 'Survived': y_pred_test})

output.to_csv('NLP_Disaster_tweet_class.csv', index=False)

print("Your submission was successfully saved!")