from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
%matplotlib inline
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd #data processing

import os
import re
import nltk
train=pd.read_csv('E:\\Kaggle\\Disaster Tweets\\train.csv')
test=pd.read_csv('E:\\Kaggle\\Disaster Tweets\\test.csv')
print(train.shape, test.shape)
print(train.isnull().sum())
print('************')
print(test.isnull().sum())
test=test.fillna(' ')
train=train.fillna(' ')
real_words = ''
fake_words = ''
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in train[train['target']==1].text: 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    real_words += " ".join(tokens)+" "

for val in train[train['target']==0].text: 
      
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    fake_words += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(real_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(fake_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#Cleaning and preprocessing
stop_words = stopwords.words('english')
lemmatizer=WordNetLemmatizer()
for index,row in train.iterrows():
    filter_sentence = ''
    
    sentence = row['text']
    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning
    
    words = nltk.word_tokenize(sentence) #tokenization
    
    words = [w for w in words if not w in stop_words]  #stopwords removal
    
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    train.loc[index,'text'] = filter_sentence
train = train[['text','target']]
#NLP Techniques
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
X_train = train['text']
Y_train = train['target']
#Feature extraction using count vectorization and tfidf.
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
tf_idf_matrix
test_counts = count_vectorizer.transform(test['text'].values)
test_tfidf = tfidf.transform(test_counts)

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, random_state=0)
#Multinomial Bayes
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
pred = NB.predict(X_test)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
pred1 = knn.predict(X_test)
print('Accuracy of KNN  classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred1)
cm
#Logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train, y_train)
pred2 = lr.predict(X_test)
print('Accuracy of LR  classifier on training set: {:.2f}'
     .format(lr.score(X_train, y_train)))
print('Accuracy of LR classifier on test set: {:.2f}'
     .format(lr.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred2)
cm
#SVM
from sklearn.svm import SVC
svm=SVC(kernel="linear",C=0.025,random_state=20)
svm.fit(X_train, y_train)
pred3 = svm.predict(X_test)
print('Accuracy of SVm  classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred3)
cm
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfm=RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,random_state=101,max_features = None,min_samples_leaf = 30)
rfm.fit(X_train, y_train)
pred4 = rfm.predict(X_test)
print('Accuracy of Random Forest  classifier on training set: {:.2f}'
     .format(rfm.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(rfm.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred4)
cm
#Decison Trees
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=10, random_state=101,max_features = None, min_samples_leaf = 15)
dtree.fit(X_train, y_train)
pred5 = dtree.predict(X_test)
print('Accuracy of Random Forest  classifier on training set: {:.2f}'
     .format(dtree.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(dtree.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred5)
cm