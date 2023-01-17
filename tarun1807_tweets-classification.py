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
tweet_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
tweet_test
tweet_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
tweet_train
tweet_train.describe()
tweet_train.isna().sum()
tweet_train.shape
#tweet_train['target'].value_counts()
tweet_train['target'].value_counts().plot.bar()
new_keyword  = tweet_train[tweet_train['keyword'].notna()]
new_keyword
new_keyword['keyword'].value_counts()
old_keyword  = tweet_train[tweet_train['keyword'].isna()]
old_keyword
tweet_train['keyword'] = tweet_train['keyword'].str.replace(r"[^a-zA-Z]+", " ").str.strip()
tweet_train['keyword']
keywords = new_keyword['keyword'].tolist()
keywords

old_keyword['newcol'] = old_keyword['text'].str.findall('|'.join(keywords)).apply(set).str.join(', ')
old_keyword['newcol']
old_keyword['count'] = old_keyword['newcol'].str.count(' ') + 1
old_keyword
greaterthan1 = old_keyword[old_keyword['count']>1]
greaterthan1
old_keyword['text_new'] = old_keyword['newcol'].str.split(',').str[0]
old_keyword['text_new'].isna().sum()
old_keyword
old_keyword.drop(['newcol'],inplace = True, axis =1)
old_keyword.drop(['count'],inplace = True, axis =1)
old_keyword
old_keyword['keyword'] = old_keyword['text_new']
old_keyword.drop(['text_new'],inplace = True, axis =1)
old_keyword
new_data = new_keyword.merge(old_keyword, how='outer')
locate = new_data['location'].value_counts().head(10)
locate.plot.bar()
col_one_list = tweet_train['keyword'].unique().tolist()
col_one_list
tweet_train["text"] = tweet_train["text"].str.replace(r"[^a-zA-Z]+", " ").str.strip()
tweet_test["text"] = tweet_test["text"].str.replace(r"[^a-zA-Z]+", " ").str.strip()
tweet_train["text"]
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

tweet_train["text"] = tweet_train["text"].apply(lemmatize_text)
tweet_test["text"] = tweet_test["text"].apply(lemmatize_text)
tweet_train["text"]

from nltk.corpus import stopwords
tweet_train["text"]= tweet_train["text"].apply(' '.join)
tweet_train["text"] = tweet_train["text"].str.lower().str.split()
tweet_test["text"]= tweet_test["text"].apply(' '.join)
tweet_test["text"] = tweet_test["text"].str.lower().str.split()
stop = stopwords.words('english')
tweet_train["text"] = tweet_train["text"].apply(lambda x: [item for item in x if item not in stop])
tweet_train["text"] = tweet_train["text"].apply(' '.join)
tweet_test["text"] = tweet_test["text"].apply(' '.join)
tweet_train["text"]
tweet_train['text'].head()
# Applying TF-IDDF Vectorisation on the Short description column to convert the string values in vectors form. 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df= 2, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(tweet_train["text"]).toarray()
final_features.shape
# Applying TF-IDDF Vectorisation on the Short description column to convert the string values in vectors form. 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df= 2, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(tweet_test["text"]).toarray()
final_features.shape
#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
X = tweet_train["text"]
Y = tweet_train['target']
test_head = tweet_test["text"]
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k='all')),
                     ('clf', RandomForestClassifier())])
# fitting our model and save it in a pickle for later use
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
model = pipeline.fit(X, Y)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)
#ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
#print(classification_report(ytest, model.predict(X_test)))
#print(confusion_matrix(ytest, model.predict(X_test)))

# Predicting the Main Category column as per the ML algorithm.
test_data = model.predict(test_head)
tweet_test['target'] = test_data
tweet_test
columns = ['id','target']
submission = tweet_test[columns]
submission