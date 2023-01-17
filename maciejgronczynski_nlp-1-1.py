# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib
import matplotlib.pyplot as plt

#SKLEARN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#OTHER LIBRARIES
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import string
import re
import nltk





# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Loading Successful.')
test_data  = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
print('Loading Successful.')
train_data.head(20)

train_data.shape

train_data.groupby('target').count()
#.head(n=50) - is a great function that allows you to see 50 records not just 10
train_data.groupby('keyword').size().head(n=50)
train_data.describe()
train_data.info()
#Not working correctly x-axis should display 0 and 1 value.

train_values = train_data['target'].value_counts()
train_values = train_values[:2,]
barlist = plt.bar(train_values.index, train_values.values, alpha=0.8)
barlist[0].set_color('r')
plt.xlabel('value', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('fake news(0)         vs         real news(1)')
plt.show()



#Great loop to show full text of each tweet as .head() might not display full message beacuse of it lenght.
t = train_data["text"].to_list()
for i in range(5):
    print('Tweet Number '+str(i+1)+': '+t[i])
train_data['location'].value_counts().head(n=20)

#You need to make train_keyword a list to get number of unique values of 'keyword' or you will get total number of all records which is useless.
train_keyword = list(set(train_data['keyword']))

print(len(train_keyword))
#the way you can check if there's any duplicate messages 7613-7503 = 110 . thats the number of duplicated messages 
x1 = len(train_data) #=7613
x2 = len(set(train_data['text'])) #=7498
number_of_duplicated_records = (x1 - x2) 
print('Number of duplicated records is:',number_of_duplicated_records)
def remove_punct(text):
    text_nopunct = ''.join([char for char in text if char not in string.punctuation])
    return text_nopunct

train_data['text'] = train_data['text'].apply(lambda x: remove_punct(x))
# this is how you can drop any column. --> train_data.drop('text_clean', axis=1, inplace=True)
train_data.head()

#function to tokenize words
def tokenize(text):
    tokens = re.split('\W+',text) #W+ means that either a word character (A-Z) or a dash(-) can go there.
    return tokens

#converting to lowercase as python is case-sensitive
train_data['text'] = train_data['text'].apply(lambda x: tokenize(x.lower()))
train_data.head()
stopword = nltk.corpus.stopwords.words('english') #all english stopwords
#function to remove stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]#to remofe all stopwords
    return text

train_data['text'] = train_data['text'].apply(lambda x: remove_stopwords(x))
train_data.head()
# 1.5 pre-processing (cleaning) this is how I managed to lemmatize text(uprageded stemmer)
# u can choose stemmer or lemmatize process (lemmatize is better but longer ;)

wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

train_data['text'] = train_data['text'].apply(lambda x: lemmatizing(x))

train_data.head()
train_data.head(50)
train_data=train_data.drop('keyword',1)
train_data=train_data.drop('location',1)

test_data=test_data.drop('keyword',1)
test_data=test_data.drop('location',1)
train_data.head(50)
#MACHINE LEARNING

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(analyzer='word', binary=True)
vectorizer.fit(train_data['text'])
X = vectorizer.transform(train_data['text']).todense()
y = train_data['target'].values
X.shape, y.shape
