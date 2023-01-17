# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test.head()
print(train.shape)
print(test.shape)
train.isna().sum()
loc_na = train[train["location"].isnull()]
train.dropna(inplace=True,how="any")
train = train.reset_index()
train.isna().sum()
print(test.keyword.mode())
print(test.location.mode())
test["keyword"] = test.keyword.fillna("deluged")
test["location"] = test.location.fillna("New York")
test.isna().sum()
from sklearn.preprocessing import LabelEncoder #convert object to int
le = LabelEncoder()
train["keyword"] = le.fit_transform(train["keyword"])
train["location"] = le.fit_transform(train["location"])
test["keyword"] = le.fit_transform(test["keyword"])
test["location"] = le.fit_transform(test["location"])
import spacy
nlp = spacy.load('en')
import re
def normalize(msg):
    
    msg = str(msg)
    msg = re.sub(r"http\S+", "", msg) #remove urls
    msg = re.sub('#[^\s]+','',msg) #remove hashtags
    msg = re.sub('@[^\s]+','',msg) #remove tags
    msg = re.sub(r'[0-9]+','', msg)
    doc = nlp(msg)
    res=[]
    for token in doc:
        if(token.is_stop or token.is_punct): #word filteration
            pass
        else:
            res.append(token.lemma_.lower())
    return " ".join(res)
train["text"] = train["text"].apply(normalize)
test["text"] = test["text"].apply(normalize)
text = pd.concat([train["text"],test["text"]])
text
print(train.shape)
print(test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer #vectorize the string
c = TfidfVectorizer(ngram_range=(1,2))
mat=pd.DataFrame(c.fit_transform(text).toarray(),columns=c.get_feature_names(),index=None)
mat
train_ = train.drop(["id","text"],axis=1) 
train_ = pd.concat([train_, mat],axis=1)
train_.dropna(how = "any", inplace=True)
train_
train_["target"].iloc[:,0]
cross_val_score(LogisticRegression(), train_.drop("target", axis=1),train_["target"].iloc[:,0],cv=10).mean()
lr = LogisticRegression()
lr.fit(train_.drop(["target" ,"id"], axis=1),train_["target"].iloc[:,0])
test_ = test.drop(["text"],axis=1) 
test_ = pd.concat([test_, mat],axis=1)
test_
test_.dropna(how = "any", inplace=True)
test_
y = lr.predict(test_.drop("id", axis=1))
len(y)
test_["target"] = y
sub = test_[["id", "target"]]
sub.columns = ["id", "villi", "target"]
sub.drop("villi", axis=1,inplace=True)
sub.head()
sub.dtypes
sub["id"] = sub["id"].astype("int")
sub["target"] = sub["target"].astype("int")
sub.to_csv("submission.csv", index = False)