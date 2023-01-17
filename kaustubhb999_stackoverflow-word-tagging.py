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
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
#even perform stemming
ps = PorterStemmer() 
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE," ",text)
    text = re.sub(BAD_SYMBOLS_RE,"",text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in STOPWORDS]
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in STOPWORDS: 
            filtered_sentence.append(ps.stem(w)) 
    return " ".join(filtered_sentence)
train = pd.read_csv("/kaggle/input/stackoverflow-data/train.tsv",delimiter="\t")
train=train[0:2000]
train.head()
#for bag of keywords
final_tags_bag=["X"]
for tags in train["tags"]:
    tags=tags.replace("[","").replace("]","").replace("'","").split(",")
    for tag in tags:
        final_tags_bag.append(tag)
final_tags_bag=list(set(final_tags_bag))
final_tags_bag
def pre_proces_y_values(df):
    y_train=[]
    for tags in df:
        tags=tags.replace("[","").replace("]","").replace("'","").split(",")
        if(len(tags)<2):
            tags.append("X")
        y_train.append(tags)
    return y_train    

train['title'] = train['title'].map(lambda com : text_prepare(com))
x_train, y_train = train['title'].values, pre_proces_y_values(train['tags'].values)
mlb = MultiLabelBinarizer(classes=sorted(final_tags_bag))
y_train = mlb.fit_transform(y_train)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train).toarray()
x_train.shape
lr=LogisticRegression() 
clf = MultiOutputClassifier(estimator= LogisticRegression()).fit(x_train,y_train)
test = pd.read_csv("/kaggle/input/stackoverflow-data/validation.tsv",delimiter="\t") 
x_test=test["title"]
test['title'] = test['title'].map(lambda com : text_prepare(com))
x_test = vectorizer.transform(x_test).toarray()
test.head()
mlb.inverse_transform(clf.predict([x_test[2]]))