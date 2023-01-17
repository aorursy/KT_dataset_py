# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
#Reading the dataset
alexa=pd.read_csv("../input/amazon_alexa.tsv",sep="\t")
alexa.head()
alexa.info()
alexa.describe()
alexa.feedback.value_counts()
#Percentage of people who liked and disliked Alexa.
alexa.groupby("feedback").rating.count().plot(kind="pie",shadow=True, autopct='%1.1f%%',explode=(0.1,0.1));
#Length of reviews given by people
alexa["length"]=alexa.verified_reviews.apply(len)
alexa.head()
plt.figure(figsize=(8,5))
alexa.length.plot(kind="box")
#Who expressed their feelings better. Either sad people or happy people?
alexa.groupby("feedback").length.mean().plot(kind="bar");
plt.title("Average word length by both happy and unhappy people");
alexa.groupby("rating").length.mean().plot(kind="bar");
plt.title("rating vs length");
#Ratings distribution
alexa.groupby("rating").feedback.count().plot(kind="pie",shadow=True,autopct='%1.1f%%',explode=(0.1,0.1,0.1,0.1,0.1))
#Positive words
good=alexa[alexa.feedback==1].verified_reviews.unique().tolist()
good=" ".join(good)
from wordcloud import WordCloud
cv=WordCloud().generate(good)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)
#Negative reviews
bad=alexa[alexa.feedback==0].verified_reviews.unique().tolist()
bad=" ".join(bad)
from wordcloud import WordCloud
cv=WordCloud().generate(bad)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)
#Comparing both wordcloud and creating a list of words that occur in both oftenly including stopwords
common=["Amazon","device","Alexa","one","Echo","work","product"]
from nltk.corpus import stopwords
stop=stopwords.words("english")
stop.extend(["Amazon","device","Alexa","one","Echo","work","product","amazon","alexa","thing","echo","dot","use"])
#Converting to lower case
alexa.verified_reviews=alexa.verified_reviews.str.lower()
#Removing special characters ("[^a-z]">> This signifies that replace everything apart from lower case alphabets with white space)
alexa.verified_reviews=alexa.verified_reviews.str.replace("[^a-z]"," ")
#split into a list
alexa.verified_reviews=alexa.verified_reviews.str.split()
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:[word for word in x if word not in stop])
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x: " ".join(word for word in x))
#Positive words
good=alexa[alexa.feedback==1].verified_reviews.unique().tolist()
good=" ".join(good)
from wordcloud import WordCloud
cv=WordCloud().generate(good)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)
#Negative reviews
bad=alexa[alexa.feedback==0].verified_reviews.unique().tolist()
bad=" ".join(bad)
from wordcloud import WordCloud
cv=WordCloud().generate(bad)
cv
plt.figure(figsize=(10,8))
plt.imshow(cv)
alexa.verified_reviews=alexa.verified_reviews.str.split()
#Using wordlemmatizer to remove any plural word like "dogs" will become "dog"
from nltk.stem import WordNetLemmatizer
wll=WordNetLemmatizer()
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:[wll.lemmatize(word) for word in x])
#Using portstemmer to convert words to its base form
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
alexa.verified_reviews=alexa.verified_reviews.apply(lambda x:" ".join([ps.stem(word) for word in x]))
alexa.head(10)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(alexa.verified_reviews)
X=X.toarray()
y=alexa.feedback.tolist()
y=np.asarray(y)
y.shape,X.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier()
rc.fit(X_train,y_train)
y_pred_dc=dc.predict(X_test)
y_pred_rf=rc.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score_dc=accuracy_score(y_test,y_pred_dc)
from sklearn.metrics import accuracy_score
accuracy_score_rf=accuracy_score(y_test,y_pred_rf)
from sklearn import svm
sv=svm.SVC()
sv.fit(X_train,y_train)
y_pred_sv=sv.predict(X_test)
accuracy_score_sv=accuracy_score(y_test,y_pred_sv)
print("Decision Tree Accuracy=",accuracy_score_dc)
print("Random Forest=",accuracy_score_rf)
print("SVM accuracy=",accuracy_score_sv)
