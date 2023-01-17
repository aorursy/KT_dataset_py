# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x=pd.read_csv("../input/amazon.csv",header=None,encoding="latin1")
x.head()
x.info()
y=pd.read_csv("../input/imdb.csv",sep=";",header=None,encoding="latin1")

y.head()
y.info()
z=pd.read_csv("../input/yelp.csv",sep=";",header=None,encoding="latin1")
z.head()
z.info()
#concanate all the data in a variable.
data=pd.concat([x,y,z],axis=0)
data.shape
#first 5 samples 
data.head()
#Change the columns name
data.columns=["sentences","sinif"]
data.head()
data.isnull().sum()
#data information
data.info()
data.index=range(0,len(data),1)
import re
import nltk as nlp
description_list = []
for description in data.sentences:
    description = re.sub("[^a-zA-Z]"," ",description)#we remove the words "[a-zA-Z]" inside the sentence.
    description = description.lower()   # Capitalize letters  to lower case
    description = nlp.word_tokenize(description)#we divide each sentence into words
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer() # it helps us find the root of the word.
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description) #we have created a sentence by combining words
    description_list.append(description)

from sklearn.feature_extraction.text import CountVectorizer
 # the method we use to create bag of words.
max_features= 7000 #get the most used 7000 words.

#Stopwords.
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

#translates sentences into codes of 0 and 1
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  
 
y = data.iloc[:,1].values 
x = sparce_matrix
#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

#Navie Bayes Classification Algorithm
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
#Eğitim
nb.fit(x_train,y_train)
#Test
y_pred=nb.predict(x_test)
print("Navie Bayes algorithm accuracy:",nb.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb=confusion_matrix(y_test,nb.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_nb,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#KNN Classification Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print("Knn algorithm accuracy=",knn.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn=confusion_matrix(y_test,knn.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#Decision Tree Classification Algorithm
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Decision Tree algorithm accuracy=",dt.score(x_test,y_test)*100)
#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(y_test,dt.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_dt,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print("Random Forest algorithm accuracy =",rf.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf=confusion_matrix(y_test,rf.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_rf,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#Logistic Regression Classification Algorithm
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print("Logistic Regression algorithm accuracy =",lr.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr=confusion_matrix(y_test,lr.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_lr,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#Support Vector Machine Classification Algorithm
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
print("SVM algorithm accuracy",svm.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm=confusion_matrix(y_test,svm.predict(x_test))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm_svm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.figure(figsize=(24,24))

plt.suptitle("Confusion Matrixes",fontsize=24)
#Logistic Regression Confusion Matrix
plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Decision Tree Confusion Matrix
plt.subplot(2,3,2)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dt,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#K Nearest Neighbors Confusion Matrix
plt.subplot(2,3,3)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Naive Bayes Confusion Matrix
plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Random Forest Confusion Matrix
plt.subplot(2,3,5)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Support Vector Machine Confusion Matrix
plt.subplot(2,3,6)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt="d")

plt.show()

algorithms=["Navie Bayes","KNN","DecisionTree","Random Forest","Logistic Regression","Support Vector Machine"]
accuracy_score=[nb.score(x_test,y_test)*100,knn.score(x_test,y_test)*100,dt.score(x_test,y_test)*100,rf.score(x_test,y_test)*100,lr.score(x_test,y_test)*100,svm.score(x_test,y_test)*100]


trace1=go.Bar(
        x=algorithms,
        y=accuracy_score,
        name="Classification Algorithms",
        marker=dict(color="rgba(147, 255, 128, 0.5)",
                    line=dict(color="rgb(0,0,0)",width=1.5)))

data=[trace1]
layout=dict(title="Positive and Negative Sentences Classification",
            xaxis=dict(title="Classification Algorithms",ticklen=5,zeroline=False),
            yaxis=dict(title="Accuracy Score"))

fig=dict(data=data,layout=layout)
iplot(fig)