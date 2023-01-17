import pandas as pd
import numpy as np
df=pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head()
df["sentiment"]=df["sentiment"].map({"positive":1,"negative":0})
df.head()
df.shape
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
corpus=[]
for i in range(0,2000):
    review = BeautifulSoup( df['review'][i], "lxml").text
    review=re.sub("[^a-zA-z]"," ",review)
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(j) for j in review if not j in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:2000,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
classifiers = []

classifiers.append(KNeighborsClassifier())
classifiers.append(SVC())
classifiers.append(ExtraTreesClassifier())
classifiers.append(LogisticRegression())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=0.1))


cv_results = []
for classifier in classifiers :
    score=cross_val_score(classifier,X,y,cv=k_fold,n_jobs=4)
    print("for %s,Accuracy is %d:"%(classifier,round(np.mean(score)*100,2)))
    
      
