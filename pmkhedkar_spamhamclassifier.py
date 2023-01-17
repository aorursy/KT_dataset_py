import numpy as np

import pandas as pd
#loading the dataset

dataset = pd.read_csv("../input/edited-spam-ham-dataset/Dataset.csv",encoding="latin",names=['Labels','Messeges'])
#Showing few records from actual dataset

dataset.head()
dataset.columns
#check the size of dataset

dataset.shape
#check for null records

dataset.isnull().count()
#importing libraries required for visualization 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#visualizing the count of spam and normal messeges

plt.figure(figsize=(8,5))

sns.countplot(x='Labels',data=dataset)

plt.title("Number of samples of each class")
dataset['Labels'].value_counts()
#let's seperate dependent and independent varibles 

x = dataset['Messeges']

y = dataset['Labels']
#importing libraries required for data pre processing

import nltk

import re

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
#creating corpus from the dataset



def Create_Corpus(x):

  corpus = []



  for i in range(len(x)):



    #don't to do not

    message = re.sub(pattern='don\'t',repl="do not",string=x.get(i))



    #won't to will not

    message = re.sub(pattern='won\'t',repl="will not",string=message)



    #Keeping only alphabetical words, removing special characters and numbers

    message = re.sub(pattern='[^a-zA-Z]',repl=' ',string=message)



    #To Lowercase

    message = message.lower()



    #spliting the sentence in words 

    words = message.split()



    #using lemmatizer

    words = [lemmatizer.lemmatize(word) for word in words]



    sentence = ' '.join(words)

    corpus.append(sentence)

  return corpus



corpus = Create_Corpus(x)

corpus[0:5]
#Bag of words / CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=3000)

X = cv.fit_transform(corpus)
X.shape
#let's split the data into train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
x_train.shape
x_test.shape
#Building the model 

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(x_train,y_train)
#predicting test samples

y_pred = nb.predict(x_test)
#check accuracy, confusion matrix

from sklearn.metrics import accuracy_score,confusion_matrix

acc = accuracy_score(y_test,y_pred)

conf_mat = confusion_matrix(y_test,y_pred)

print("-----------Accuracy of the model-----------")

print("Accuracy : {}%".format(round(acc*100,2)))

print("Confusion Matrix : \n{}".format(conf_mat))
#Sample test example 1

a=cv.transform(["Hellow how are you i am fine"])
nb.predict(a)
#Sample test example 2

a=cv.transform(["Hi you have won lotery of 1 crore !!! congratulations"])

nb.predict(a)
#Sample test example 2

a=cv.transform(["How about word spam itself!!"])

nb.predict(a)