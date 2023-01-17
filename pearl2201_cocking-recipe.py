# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from pylab import rcParams
rcParams['figure.figsize'] = 25, 16
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
train_df.head()
print ("Train set size: ", len(train_df.index))
print ("Test set size: ", len(test_df.index))
train_df["cuisine"].unique()
train_df["cuisine"].value_counts()
sns.countplot(x="cuisine", data=train_df)
train_df["len_ingredients"] = train_df["ingredients"].apply(lambda x: len(x))
test_df["len_ingredients"] = test_df["ingredients"].apply(lambda x: len(x))
train_df.sort_values(by=["len_ingredients"],ascending=False).head(5)
print (train_df[train_df["id"] == 3885]["ingredients"].values)
print ("---------------")
print (train_df[train_df["id"] == 13430]["ingredients"].values)
train_df["len_ingredients"].describe()
sns.distplot(train_df["len_ingredients"])
sns.boxplot(y="len_ingredients",x="cuisine", data=train_df)
# split into words
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', string.punctuation)

def parse_ingredients(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    # stemming of words
    stemmed = [porter.stem(word) for word in words]
    return ' '.join(stemmed)
train_df["ingredients"] = train_df["ingredients"].apply(lambda x: [item.lower() for item in x])
test_df["ingredients"] = test_df["ingredients"].apply(lambda x: [item.lower() for item in x])
total_ingredients = []
for x in train_df["ingredients"].values:
    total_ingredients.extend(x)
for x in test_df["ingredients"].values:
    total_ingredients.extend(x)
total_ingredients = np.reshape(total_ingredients,(-1))
total_ingredients_unique = set(total_ingredients)
dict_token_word = {}
for ingredient in total_ingredients:
    dict_token_word[ingredient] = parse_ingredients(ingredient)
    
train_df["ingredients"] = train_df["ingredients"].apply(lambda x: [dict_token_word[item] for item in x])
test_df["ingredients"] = test_df["ingredients"].apply(lambda x: [dict_token_word[item] for item in x])
print (train_df[train_df["id"] == 3885]["ingredients"].values)
print ("---------------")
print (train_df[train_df["id"] == 13430]["ingredients"].values)
train_df["concat_ingredients"] = train_df["ingredients"].apply(lambda x: ', '.join(x))
test_df["concat_ingredients"] = test_df["ingredients"].apply(lambda x: ', '.join(x))
train_df.head(10)
token_ingredients = '. '.join(train_df["concat_ingredients"].values) + ". " + '. '.join(test_df["concat_ingredients"].values)
token_ingredients[:1000]
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(sublinear_tf=True, stop_words='english')
tf_idf.fit([token_ingredients])


print(tf_idf.vocabulary_)
print(tf_idf.idf_)
train_vector= tf_idf.transform(train_df["concat_ingredients"])
test_vector= tf_idf.transform(test_df["concat_ingredients"])
print ("5 first tran")
for i in range(5):
    k = train_vector[i].toarray()
    print (k[k!=0])
print ("-------------------------")
print ("5 first test")
for i in range(5):
    k = test_vector[i].toarray()
    print (k[k!=0])
train_vector= train_vector.toarray()
test_vector= test_vector.toarray()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train_df["cuisine"])
train_labels_encoded = le.transform(train_df["cuisine"])
print ("Len Set of labels: ",len(set(train_labels_encoded)))
sns.distplot(train_labels_encoded)
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte = train_test_split(train_vector,train_labels_encoded,test_size=0.15)
#from imblearn.over_sampling import SMOTE 
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_sample(xtr,ytr)
#from sklearn import linear_model
#clf = linear_model.SGDClassifier()
#clf.fit(xtr,ytr)
#clf.score(xte,yte)
#from sklearn import linear_model
#clf = linear_model.SGDClassifier()
#clf.fit(X_res,y_res)
#clf.score(xte,yte)
from sklearn.svm import LinearSVC
l_svc = LinearSVC()
l_svc.fit(xtr,ytr)
l_svc.score(xte,yte)
#from sklearn.ensemble import AdaBoostClassifier
#ada = AdaBoostClassifier()
#ada.fit(X_res,y_res)
#ada.score(xte,yte)
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier()
#rfc.fit(X_res,y_res)
#rfc.score(xte,yte)
y_pred = l_svc.predict(test_vector)
test_labels = le.inverse_transform(y_pred)

submission = pd.DataFrame({
    "Id":test_df["id"],
    "cuisine": test_labels
})
submission.to_csv("submission.csv",index=False)