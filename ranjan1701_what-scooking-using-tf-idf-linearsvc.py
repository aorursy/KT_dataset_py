import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation as cval   
import time
train = pd.read_json("../input/train.json")
print(train.head())
print (train.isnull().sum())
y = train[['cuisine']]
print(y.head())
X  = train[['ingredients']]
print(X.head())

docs = []
for ingredient in train.ingredients:
    temp = ""
    for item in ingredient:
        temp = temp + item + " "
    docs.append(temp)
print(len(docs))
#sub = "10"
#print ("\n".join(s for s in docs if sub in s))
#tokens = pd.DataFrame({"token":docs})
#tokens.to_csv("tokens1.csv",index=False)
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(docs)
#tokens = pd.DataFrame({"token":docs})
#tokens.to_csv("tokens.csv",index=False)
X_transformed.shape
target_enc = LabelEncoder()
y = target_enc.fit_transform(train.cuisine)
y = y.reshape(-1)
data_train, data_test, label_train, label_test = train_test_split(X_transformed, y, test_size=0.33, random_state=7)
data_train
label_train.shape
#data_train.describe()
label_test = label_test.reshape(-1,1)
label_test.shape
knn.score(data_test,label_test)
def tokenize(df):
    doc = []
    for ingredient in df.ingredients:
        temp = ""
        for item in ingredient:
            temp = temp + item + " "
        doc.append(temp)
    return doc
lsvc = LinearSVC()
s = time.time()
print("Training...")
lsvc.fit(data_train, label_train)
print("Training Time" + str(time.time() -s ))
s= time.time()
print("Predicting...")
print("Linear SVC Score- "+ str(lsvc.score(data_test, label_test)))
print("Predicting Time" + str(time.time() -s ))
lsvc.coef_.shape
s = time.time()
print (cval.cross_val_score(lsvc, data_train, label_train, cv=10).mean())
print("Training Time" + str(time.time() -s ))
