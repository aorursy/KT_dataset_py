import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import string



from sklearn.model_selection import train_test_split

from sklearn import preprocessing



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score



from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')
from nltk.corpus import stopwords

", ".join(stopwords.words('english'))

msg = pd.read_csv("../input/spamorham/spam.csv", encoding= "ISO-8859–1" )

msg.head(10)
msg.shape
msg.tail()
msg.groupby("Label").count()
plt.figure(figsize=[5,5])

msg["Label"].value_counts().plot(kind='pie',legend=True,cmap="Set3")

plt.ylabel("Spam/NotSpam")
msg.isna().sum()
msg["TextLength"]=msg["EmailText"].apply(len)

msg.head()
sns.set_style("darkgrid")

sns.set(rc = {'figure.figsize' : (18,6)})

msg.hist(column = 'TextLength', by = 'Label', bins = 40,edgecolor = 'black',color="orange")
plt.figure(figsize=[10,5])

plt.hist(msg[msg['Label']=='spam']['TextLength'],color='darkblue',bins=50,edgecolor='darkblue')

plt.title('Spam Message Length',fontsize=20)

plt.xlabel('Message Length')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=[10,5])

plt.hist(msg[msg['Label']=='ham']['TextLength'],color='olive',bins=50,edgecolor='olive')

plt.title('Ham Message Length',fontsize=20)

plt.xlabel('Message Length')

plt.ylabel('Count')

plt.show()
#data_preprocessesing

def preprocess(text):

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return " ".join(text)





def stem(text):

    text = text.split()

    words = ""

    for i in text:

            stemmer = SnowballStemmer("english")

            words += (stemmer.stem(i))+" "

    return words
msg["ProcessedEmailText"]=msg["EmailText"].apply(preprocess)

msg["ProcessedEmailText"]=msg["ProcessedEmailText"].apply(stem)

msg.head()
le = preprocessing.LabelEncoder()

msg["LabelNum"]=le.fit_transform(msg["Label"])
msg["ProcessedEmailText"]
cv = CountVectorizer("english")

spam_model_cnt = cv.fit_transform(msg["ProcessedEmailText"])

spam_model_cnt.shape
tf = TfidfVectorizer("english")

spam_model_tfidf = tf.fit_transform(msg["ProcessedEmailText"])

spam_model_tfidf.shape

vect_tfidf=spam_model_tfidf.toarray()

vect_cnt=spam_model_cnt.toarray()
feature=pd.DataFrame(vect_tfidf,columns=tf.get_feature_names())

feature["TextLen"]=msg["TextLength"]

feature.head()
msg_train,msg_test,ans_train,ans_test=train_test_split(feature,msg["LabelNum"],test_size=0.2,random_state=20)
#LogisticRegression

from sklearn.linear_model import LogisticRegression



model=LogisticRegression()

model.fit(msg_train, ans_train)

predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
cnf_matrix = confusion_matrix(predicted, ans_test, labels=[1,0])

cnf_matrix
print (classification_report(predicted, ans_test))
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
model=LogisticRegression()

model_gs=GridSearchCV(model,grid,cv=5,n_jobs=-1,verbose=1)

model_gs.fit(msg_train, ans_train)
model_gs.best_params_
model_gs.best_score_
#Decision_Tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

model=DecisionTreeClassifier()

model.fit(msg_train,ans_train)
predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
cnf_matrix = confusion_matrix(predicted, ans_test, labels=[1,0])

cnf_matrix
array=[1,2,3,4,5,6,7,8,9,10]

grid={'criterion':['gini','entropy'], 'max_depth':array}
model=DecisionTreeClassifier()

model_gs=GridSearchCV(model,grid,cv=5,n_jobs=-1,verbose=1)

model_gs.fit(msg_train, ans_train)
model_gs.best_params_
model_gs.best_score_
#KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(msg_train,ans_train)
predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
print (classification_report(predicted, ans_test))
array=[1,2,3,4,5,6,7,8,9,10]

grid={'n_neighbors':array, 'weights':['uniform', 'distance'], 'metric':['euclidean','manhattan','minkowski']}
model=KNeighborsClassifier()

model_gs=GridSearchCV(model,grid,verbose=1,cv=5,n_jobs=-1)

model_gs.fit(msg_train, ans_train)
model_gs.best_params_
model_gs.best_score_
#Naive_Bayes

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB(0.1)

model.fit(msg_train,ans_train)
predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
print (classification_report(predicted, ans_test))
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()

model.fit(msg_train,ans_train)
predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
print (classification_report(predicted, ans_test))
#SVM

from sklearn import svm

model=svm.SVC(C=100, gamma=0.001, kernel='rbf')

model.fit(msg_train,ans_train)
predicted=model.predict(msg_test)

accuracy_score(predicted,ans_test)
print (classification_report(predicted, ans_test))