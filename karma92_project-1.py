#Import libraries

import numpy as np 

import pandas as pd 

import nltk

from nltk.corpus import stopwords

import string
import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read csv file & assign to a dataframe from https://www.kaggle.com/karthickveerakumar/spam-filter/version/1#emails.csv

df = pd.read_csv("../input/spam-filter/emails.csv")
#show the first 5 elements of the csv

df.head(10)
#print number of rows and columns

df.shape
df.info()
#show element n

print(df.iloc[100][0])

df.isnull().sum() #shows how many rows are null, this is good practice if we want to take care of null value in data prep
#In Python3, string.punctuation is a pre-initialized string used as string constant, this contain all the punctuation

punct= string.punctuation
punct
#nltk.download('stopwords')

#this package from natural language toolkit contains all the stopword in the major languages
stop_words = set(stopwords.words("italian"))
stop_words
#now we remove all punctuations and stop words

df
def remove_punct(text):

    nopunct = [char for char in text if char not in punct]

    nopunct = ''.join(nopunct)

    return(nopunct)



#this function remove the punctuation

    
df1 = df['text'].apply(remove_punct)
df1
#now we remove all stop words

def remove_stop(text):

    nostop = [word for word in text.split() if word.lower() not in stop_words]

    return(nostop)
df2 = df1.apply(remove_stop)
df2 #at this point we have a list of tokens to analyze
def remove_everything(text):

    return remove_stop(remove_punct(text))



#this is definetely not the best coding - but is just to show the progression
df['text'].apply(remove_everything)
from sklearn.feature_extraction.text import CountVectorizer

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
message = 'KEA is a university, /////located in Nørrebro I love burger burger'

message2 = "are great or burger, burger"

message3 = 'hey i love berries and burger love burger a lot burger'

print(message)
vectorizer = CountVectorizer(analyzer=remove_everything)

#initialize a vector object

vect_fit_trans = vectorizer.fit_transform([message,message2,message3]) 

#run the function fit_transform over the array of messages
vectorizer.get_feature_names()

#shows all the features extracted - they are simply put in alphabetical order
print(vect_fit_trans)

#each word is transformed into a vector (x,y) n

#bag of words
transformvc = CountVectorizer(analyzer=remove_everything)
FiTrannsformvc = transformvc.fit_transform(df['text'])
transformvc.get_feature_names()

#now we split the dataset 80% for training and 20% for test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(FiTrannsformvc, df['spam'], test_size=0.20, random_state=75)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)



# I used Naive Bayes, being one of the most popular algoryth for spam detection

# https://www.quora.com/What-are-the-popular-ML-algorithms-for-email-spam-detection
X_train.shape #verify that train & test have the same shape, we fit_transform before the split, therefore it shouldn't be a problem
X_test.shape
print(classifier.predict(X_train))

print(y_train.values)
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

pred = classifier.predict(X_train)

print(classification_report(y_train ,pred ))

print('Confusion Matrix: \n',confusion_matrix(y_train,pred))

print()

print('Accuracy: ', accuracy_score(y_train,pred))
import seaborn as sn

import matplotlib.pyplot as plt
#Print the predictions

print('Predicted value: ',classifier.predict(X_test))



#Print Actual Label

print('Actual value: ',y_test.values)
#here we evaluate the model on the test data set

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

pred = classifier.predict(X_test)

print(classification_report(y_test ,pred ))





print('Confusion Matrix: \n', confusion_matrix(y_test,pred))

print()

print('Accuracy: ', accuracy_score(y_test,pred))
type(X_test)

X_test.shape

realmsg = '''

INSERT A MESSAGE HERE

'''

print(realmsg)
realvector = transformvc.transform([realmsg])
type(realvector)

realvector.shape

confidence = classifier.predict_proba(realvector)

prediction = classifier.predict(realvector)

print(confidence[0,1])
if confidence[0,1] > 0.8:

    print("This e-mail is Spam")

elif (confidence[0,1] > 0.5) and (confidence[0,1] < 0.8):

    print("This e-mail seems to be Spam, check it")

else:

    print("This e-mail is legit")
print(prediction)