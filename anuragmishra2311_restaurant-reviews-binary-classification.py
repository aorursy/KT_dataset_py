import pandas as pd
import numpy as np 
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('../input/restaurant-reviews/Restaurant_Reviews.csv')
df.head()
from nltk.corpus import stopwords

# xyz is a list consisting of English Stopwords
xyz=stopwords.words('english') 
xyz.remove('not')
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df.iloc[i,0]) # Removing punctuations
    review=review.lower() # Converting to lower case.
    review=review.split() # List of words in a review.
    ps=PorterStemmer() # Stemming Words
    review=[ps.stem(word) for word in review if not word in set(xyz)] #Stemming words those are not in list xyz i.e list of stopping words
    review=' '.join(review)
    corpus.append(review)
    
original=list(df.Review)
original[:10]
corpus[0:10]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
no_of_words=cv.fit_transform(corpus).toarray()
len(no_of_words[0])
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
print('Test Score : {} %'.format(classifier.score(X_test,y_test)*100))
cross_val_score(classifier,X_train,y_train,cv=10).mean()*100
skplt.metrics.plot_confusion_matrix(y_test,classifier.predict(X_test),figsize=(8,8))
logistic_classifier=LogisticRegression()
logistic_classifier.fit(X_train,y_train)
print('Test Score : {} %'.format(logistic_classifier.score(X_test,y_test)*100))
cross_val_score(logistic_classifier,X_train,y_train,cv=10).mean()*100
skplt.metrics.plot_confusion_matrix(y_test,logistic_classifier.predict(X_test),figsize=(8,8))