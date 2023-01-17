# importing all required libraries
import numpy as np 
import pandas as pd
import os
import nltk
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn import metrics

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


A = pd.read_csv('../input/spam.csv', encoding = 'latin-1')
A.head()
A = A.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
# function to clean text. It performs following function:
#            1) converts all the SMS tex t to lower case.
#            2) removes any website link.
#            3) removes any kind of punctuations from text. Punctuations in string.punctuation are listed below.
#               {';', "'", '%', '~', '.', '_', '<', '/', '`', ')', '\\', ']', '&', '#', ',', '^', 
#                ':', '>', '-', '$', '!', '{', '@', '(', '*', '"', '|', '=', '+', '?', '}', '['}
#            4) removes numbers from the text.
#            5) removes single letters.
#            6) removes stop words from the text.
#                to check the list of stop words run the following command : print(set(stopwords.words('english')))
#            7) Stem the words.stemming work by cutting off the end or the beginning of the word, taking into account 
#               a list of common prefixes and suffixes that can be found in an inflected word.

# finaly the words are again rejoined to make sentence after cleaning

def SMSpreprocess_func(A):
    temp1 =[]
    port = PorterStemmer()
    for text in A:
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r'['+string.punctuation+']', r'', text)
        text = re.sub('\d+','',text)
        text = ' '.join( [w for w in text.split() if len(w)>1] )
        
        words = [port.stem(word) for word in text.split() if word not in stopwords.words('english')]
        temp1.append(words)
        
    
    temp2 = []
    for row in temp1:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        temp2.append(sequ)
    
    
    return temp2

# function which takes train samples (X_train and Y_train)to train the model and 
# test samples to test and gives out the predicted output array  

def predict_func(classifier, X_train,X_test,Y_train):
    classifier.fit(X_train,Y_train)
    predict = classifier.predict(X_test)
    return predict

# following functions calculate accuracy, confusion matrix, precision(for spam) and recall(for spam) for model evaluation
def accuracy_func(predict,Y_test):
    accuracy = metrics.accuracy_score(predict,Y_test)
    return accuracy

def ConfusionMatrix_func(predict,Y_test):
    c = metrics.confusion_matrix(Y_test,predict)
    return c

def precision_func(c):
    precision = c[1,1]/(c[1,1] + c[0,1])
    return precision

def recall_func(c):
    recall = c[1,1]/(c[1,1] + c[1,0])
    return recall
#cleaning and processing SMS data
X_clean = SMSpreprocess_func(A.v2)

#label encoding the categorical dependent variable; Spam as 1 and ham as 0
encoder = LabelEncoder()
encoder.fit(['spam','ham'])
Y = list(encoder.transform(A.v1))
# performing holdout cross-validation to split the data into train(75%) and test(25%) sample.
X_train,X_test,Y_train,Y_test = train_test_split(X_clean,Y,test_size = 0.25, random_state = 42)

#feature extraction: Feature extraction is transforming arbitrary data, such as text or images, into numerical features
#usable for machine learning.
#tokenizing and counting the words; created a sparse matrix that can be used to train the model
count_vect = CountVectorizer()
count_vect_ngram = CountVectorizer(ngram_range = (2,3))

count_vect.fit(X_train)
count_vect_ngram.fit(X_train)

# tranforming words to array of numbers
X_traincount = count_vect.transform(X_train)
X_testcount = count_vect.transform(X_test)

X_train_ngram = count_vect_ngram.transform(X_train)
X_test_ngram = count_vect_ngram.transform(X_test)
# different classifiers that are going to be used train the models.
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = naive_bayes.MultinomialNB()
knn = KNeighborsClassifier()
spam_classifier = { 'Logistic Regression' : lr, 'Decision Tree': dt, 'Random Forest': rf, 'Naive Bayes' : nb, 'KNN': knn}
summary = []
for n,c  in spam_classifier.items():
    predict = predict_func(c,X_traincount,X_testcount,Y_train)
    accuracy = accuracy_func(predict,Y_test)
    confusion_matrix = ConfusionMatrix_func(predict,Y_test)
    precision = precision_func(confusion_matrix)
    recall = recall_func(confusion_matrix)
    summary.append((n,[accuracy, precision, recall]))
    print("Confusion Matrix of",n, "\n", confusion_matrix)

result_summary =  pd.DataFrame.from_items(summary,orient='index', columns=['Accuracy', 'Precision', 'Recall'])
result_summary
#Although all the models gives an accuracy above 90%, we cannot make the evaluation of the model using accuracy because:
# 1) In spam detection we always try to increase the prediction of spam messages but more importantly we also dont want ham to be
#labelled as spam and end up in junk folder.
#2) In problems like spam detection, fraud detection there exist a class imbalance problem so looking at the accuracy of the model can
#mislead us.

#To overcome the above mentioned problem we look into confusion matrix and precision,recall rate of spam class.
# so Naive bayes seems to perform better than other methods.

summary_ngram = []
for n,c  in spam_classifier.items():
    predict = predict_func(c,X_train_ngram,X_test_ngram,Y_train)
    accuracy = accuracy_func(predict,Y_test)
    confusion_matrix = ConfusionMatrix_func(predict,Y_test)
    precision = precision_func(confusion_matrix)
    recall = recall_func(confusion_matrix)
    summary_ngram.append((n,[accuracy, precision, recall]))
    print("Confusion Matrix of" , n, "\n", confusion_matrix)

result_summary_ngram =  pd.DataFrame.from_items(summary_ngram,orient='index', columns=['Accuracy', 'Precision', 'Recall'])
print(result_summary_ngram)
#still Naive bayes seems to perform better than other model