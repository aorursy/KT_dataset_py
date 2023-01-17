import pandas as pd
import numpy as np
#Set current working directory
import os
os.getcwd()
os.chdir('../input/textdb3/')
#Importing the dataset from directory
dataset = pd.read_csv('fake_or_real_news.csv')
subdataset = dataset[['title','text']]
subdataset = np.array(subdataset)
#We will use Natural Language Processing to clean the text to train our model
#Cleaning the Texts
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus = []    #This list will contain all essential words required to train the model
for i in range(0, 6335):
    review = re.sub('[^a-zA-Z]', ' ', str(subdataset[i, :]).strip('[]')) #Need to remove all non-essential and non-English characters.
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #Need to stem the words to reduce number of words, we remove all stopwords or grammer words that are used form a sentence.
    review = ' '.join(review)
    corpus.append(review)
#Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500) #This will create a sparse matrix of the top 2500 words that occur in a particular row.
X = cv.fit_transform(corpus).toarray() #Our Dependent variable containing the title and text corpus 
y = dataset.iloc[:, 3].values  #Our Independent Varaible containing the labels: REAL or FAKE
#We'll use Naive Bayes Classification
#Splitting dataset into Test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state = 0)
#Fitting Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#Predicting on validation set
y_pred = classifier.predict(X_test)
#Accuracy Score and Confusion Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
#K-Fold Validation for complete evalution 
from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
kcv.mean()
kcv.std()