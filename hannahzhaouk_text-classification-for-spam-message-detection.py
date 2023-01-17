#import useful packages
import numpy as np
import pandas as pd
#download nltk's 'stopwords' for removal in pre-process
import nltk
nltk.download('stopwords')
#read data, rename the columns, and check the format
SMS= pd.read_csv('../input/filtering-mobile-phone-spam/sms_spam.csv',  names=['label','messages'] )
SMS.head()
SMS=SMS.iloc[1:]  #remove original labels
SMS.head()
SMS.groupby('label').describe() #explore the data to get some initial understanding
#next step is to pre-process text data 

#firstly import useful functions
import string
from nltk.corpus import stopwords

#    define a function to:
#    1. Remove punctuations
#    2. tokenize the terms in each text message
#    3. Remove stopwords

def text_process(content):
    removepunc=[word for word in content if word not in string.punctuation]
    removepunc=''.join(removepunc)
    
    return[term for term in removepunc.split() if term.lower() not in stopwords.words('english')]

SMS['messages'].head(5).apply(text_process) #check if defined function works 
#Next step is to vectorize each term and weight it by tf-idf model

#firstly import cpuntvectorizer to measure the frequency of each word term

from sklearn.feature_extraction.text import CountVectorizer 
bow_process=CountVectorizer(analyzer=text_process).fit(SMS['messages'])

print (len(bow_process.vocabulary_)) #check the number of terms (vectors)
#transform vectors to term-document incidence matrix
SMS_bow= bow_process.transform(SMS['messages'])
print('Shape :', SMS_bow.shape) #check the size of term-document incidence matrix
#weight vectors by tf-idf model
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_trans=TfidfTransformer().fit(SMS_bow)
SMS_tfidf= tfidf_trans.transform(SMS_bow)

print(SMS_tfidf.shape) #check the size of weighted term-document incidence matrix
from sklearn.naive_bayes import MultinomialNB
naive_bayes_model=MultinomialNB().fit(SMS_tfidf, SMS['label']) #train Naive Bayes classifier
from sklearn.svm import NuSVC
SVM_model=NuSVC(nu = 0.05, class_weight = 'balanced').fit(SMS_tfidf, SMS['label']) #train SVM classifier
from sklearn.neighbors import KNeighborsClassifier
KNN_model=KNeighborsClassifier().fit(SMS_tfidf, SMS['label']) #train KNN classfier
#now we have developed three trained classification models
#next step is to test and evaluate the models
#first need to use the same way to pre-process test data to weighted vectors
SMS_test= pd.read_csv('../input/spam-test-set/spam_test.csv')
SMS_test=SMS_test.rename(columns={'v1':'label','v2':'messages'})
SMS_test=SMS_test.iloc[:,:2]
SMS_test.head()
SMS_test['messages'].head(5).apply(text_process)
SMS_test_bow=bow_process.transform(SMS_test['messages'])
SMS_test_tfidf=tfidf_trans.transform(SMS_test_bow) #now we have vectorized test data that can be classified by three models
print(SMS_test_bow.shape)
from sklearn.metrics import classification_report
NB_predict = naive_bayes_model.predict(SMS_test_tfidf)    # test the Naive Bayes model and get prediction

print(classification_report(SMS_test['label'],NB_predict))    # generate evaluation report of NB model
SVM_predict=SVM_model.predict(SMS_test_tfidf)    # test the SVM model and get prediction
print(classification_report(SMS_test['label'],SVM_predict))     # generate evaluation report of SVM model
KNN_predict=KNN_model.predict(SMS_test_tfidf)     # test the KNN model and get prediction
print(classification_report(SMS_test['label'],KNN_predict))     # generate evaluation report of KNN model
#create pipelines for three models to systematically pre-process text data based on our previous pre-processing steps
#to store pipelines of workfolow
#for further study use

from sklearn.pipeline import Pipeline
NB_classifier = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),   #vectorize terms within text data sets
    ('tfidf', TfidfTransformer()),  #weight terms
    ('classifier', MultinomialNB()),   #implement Naive Bayes classifier
])
NB_classifier.fit(SMS['messages'],SMS['label'])  #train the model by fitting training data sets
NB_prediction=NB_classifier.predict(SMS_test['messages'])  #test the model and get prediction
print(classification_report(SMS_test['label'],NB_prediction))  #create evaluation report
### same steps as above to produce ###

SVM_classifier = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)), 
    ('tfidf', TfidfTransformer()),   
    ('classifier', NuSVC(nu = 0.05, class_weight = 'balanced')),
])
SVM_classifier.fit(SMS['messages'],SMS['label'])
SVM_prediction=SVM_classifier.predict(SMS_test['messages'])
print(classification_report(SMS_test['label'],SVM_prediction))
KNN_classifier= Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)), 
    ('tfidf', TfidfTransformer()),   
    ('classifier', KNeighborsClassifier()),
])
KNN_classifier.fit(SMS['messages'],SMS['label'])
KNN_prediction=KNN_classifier.predict(SMS_test['messages'])
print(classification_report(SMS_test['label'],KNN_prediction))
