import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import glob

import time

import pandas as pd



from nltk import ngrams

from nltk.tokenize import sent_tokenize

import nltk

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

from nltk.stem import PorterStemmer

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize
import pandas as pd

df = pd.read_csv("../input/sql-injection-dataset/sqli.csv",encoding='utf-16')




from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))

posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()



transformed_posts=pd.DataFrame(posts)
df=pd.concat([df,transformed_posts],axis=1)
X=df[df.columns[2:]]
y=df['Label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X.head()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)
from keras.models import Sequential

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.wrappers.scikit_learn import KerasClassifier
input_dim = X_train.shape[1]  # Number of features



model = Sequential()

model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(10,  activation='tanh'))

model.add(layers.Dense(1024, activation='relu'))



model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

model.summary()
classifier_nn = model.fit(X_train,y_train,

                    epochs=10,

                    verbose=True,

                    validation_data=(X_test, y_test),

                    batch_size=15)

pred=model.predict(X_test)
for i in range(len(pred)):

    if pred[i]>0.5:

        pred[i]=1

    elif pred[i]<=0.5:

        pred[i]=0
accuracy_score(y_test,pred)
def accuracy_function(tp,tn,fp,fn):

    

    accuracy = (tp+tn) / (tp+tn+fp+fn)

    

    return accuracy
def precision_function(tp,fp):

    

    precision = tp / (tp+fp)

    

    return precision
def recall_function(tp,fn):

    

    recall=tp / (tp+fn)

    

    return recall
def confusion_matrix(truth,predicted):

    

    true_positive = 0

    true_negative = 0

    false_positive = 0

    false_negative = 0

    

    for true,pred in zip(truth,predicted):

        if true == 1:

            if pred == true:

                true_positive += 1

            elif pred != true:

                false_negative += 1



        elif true == 0:

            if pred == true:

                true_negative += 1

            elif pred != true:

                false_positive += 1

            

    accuracy=accuracy_function(true_positive, true_negative, false_positive, false_negative)

    precision=precision_function(true_positive, false_positive)

    recall=recall_function(true_positive, false_negative)

    

    return (accuracy,

            precision,

           recall)
accuracy,precision,recall=confusion_matrix(y_test,pred)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
from sklearn.metrics import precision_score

precision_score(y_test, pred)
from sklearn.metrics import recall_score

recall_score(y_test, pred)
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()



gnb.fit(X_train, y_train)



pred_gnb = gnb.predict(X_test)
# SVM

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

pred_svm=clf.predict(X_test)
# KNN

from sklearn.neighbors import KNeighborsClassifier



neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)



pred_knn = neigh.predict(X_test)
# DT

from sklearn import tree



dt = tree.DecisionTreeClassifier()

dt = dt.fit(X_train, y_train)



pred_dt = dt.predict(X_test)
def accuracy_function(tp,tn,fp,fn):

    

    accuracy = (tp+tn) / (tp+tn+fp+fn)

    

    return accuracy



def precision_function(tp,fp):

    

    precision = tp / (tp+fp)

    

    return precision



def recall_function(tp,fn):

    

    recall=tp / (tp+fn)

    

    return recall



def confusion_matrix(truth,predicted):

    

    true_positive = 0

    true_negative = 0

    false_positive = 0

    false_negative = 0

    

    for true,pred in zip(truth,predicted):

        

        if true == 1:

            if pred == true:

                true_positive += 1

            elif pred != true:

                false_negative += 1



        elif true == 0:

            if pred == true:

                true_negative += 1

            elif pred != true:

                false_positive += 1

            

    accuracy = accuracy_function(true_positive, true_negative, false_positive, false_negative)

    precision = precision_function(true_positive, false_positive)

    recall = recall_function(true_positive, false_negative)

    

    return (accuracy,precision,recall)
accuracy,precision,recall=confusion_matrix(y_test,pred)

print(" For CNN \n Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
accuracy,precision,recall=confusion_matrix(y_test,pred_gnb)

print(" For Naive Bayes Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
accuracy,precision,recall=confusion_matrix(y_test,pred_svm)

print(" For SVM Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
accuracy,precision,recall=confusion_matrix(y_test,pred_knn)

print(" For KNN Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
accuracy,precision,recall=confusion_matrix(y_test,pred_dt)

print(" For Decision Tree Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
from keras.models import load_model

import pickle



model.save('my_model_cnn.h5')

with open('vectorizer_cnn', 'wb') as fin:

    pickle.dump(vectorizer, fin)



import keras

from keras.models import load_model

import pickle

import tensorflow as tf





mymodel = tf.keras.models.load_model('my_model_cnn.h5')

myvectorizer = pickle.load(open("vectorizer_cnn", 'rb'))









def clean_data(input_val):



    input_val=input_val.replace('\n', '')

    input_val=input_val.replace('%20', ' ')

    input_val=input_val.replace('=', ' = ')

    input_val=input_val.replace('((', ' (( ')

    input_val=input_val.replace('))', ' )) ')

    input_val=input_val.replace('(', ' ( ')

    input_val=input_val.replace(')', ' ) ')

    input_val=input_val.replace('1 ', 'numeric')

    input_val=input_val.replace(' 1', 'numeric')

    input_val=input_val.replace("'1 ", "'numeric ")

    input_val=input_val.replace(" 1'", " numeric'")

    input_val=input_val.replace('1,', 'numeric,')

    input_val=input_val.replace(" 2 ", " numeric ")

    input_val=input_val.replace(' 3 ', ' numeric ')

    input_val=input_val.replace(' 3--', ' numeric--')

    input_val=input_val.replace(" 4 ", ' numeric ')

    input_val=input_val.replace(" 5 ", ' numeric ')

    input_val=input_val.replace(' 6 ', ' numeric ')

    input_val=input_val.replace(" 7 ", ' numeric ')

    input_val=input_val.replace(" 8 ", ' numeric ')

    input_val=input_val.replace('1234', ' numeric ')

    input_val=input_val.replace("22", ' numeric ')

    input_val=input_val.replace(" 8 ", ' numeric ')

    input_val=input_val.replace(" 200 ", ' numeric ')

    input_val=input_val.replace("23 ", ' numeric ')

    input_val=input_val.replace('"1', '"numeric')

    input_val=input_val.replace('1"', '"numeric')

    input_val=input_val.replace("7659", 'numeric')

    input_val=input_val.replace(" 37 ", ' numeric ')

    input_val=input_val.replace(" 45 ", ' numeric ')



    return input_val

















def predict_sqli_attack():

    

    repeat=True

    

    beautify=''

    for i in range(20):

        beautify+= "*"



    print(beautify) 

    input_val=input("Enter a sentence : ")





    

    if input_val== '0':

        repeat=False

    

    



    input_val=clean_data(input_val)

    input_val=[input_val]







    input_val=myvectorizer.transform(input_val).toarray()

    

   # input_val.shape=(1,64,64,1)



    result=mymodel.predict(input_val)





#     print(beautify

    

    

    if repeat == True:

        

        if result>0.5:

            print("ALERT!!!! SQL injection Detected")





        elif result<=0.5:

            print("It is normal")

            

        print(beautify)

            

        predict_sqli_attack()

            

    elif repeat == False:

        print( " Closing detection ")



 



# Uncomment the function call below and enter the strings to detect the SQL injection attack



# predict_sqli_attack()