import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter='\t' )
dataset.sample(5)
import re
review=re.sub('[^a-zA-Z]',' ', dataset.Review.iloc[5])
review
review=review.lower()
review
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
review=review.lower()
review=review.split()
review=[word for word in review]

corpus=[]
dataset.shape
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
y.shape
X.shape
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
clf=GaussianNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm
x=sum(sum(cm))
acc=(cm[0][0]+cm[0][1])/x
sum(cm)[0]
sum(cm)[1]
sum(cm)
pre=cm[0][0]/sum(cm)[0]
rec=cm[0][1]/sum(cm)[1]
F_score=2*pre*rec/(pre+rec)
F_score