# To read the csv files in arrays and dataframes.

import numpy as np 

import pandas as pd 
data = pd.read_csv("../input/spam.csv", encoding = "latin-1")

# # encoding='latin-1' is used to download all special characters and everything in python. If there is no encoding on the data, it gives an error. Let's check the first five values.

data.head()
data.isnull().sum()
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

data.rename(columns= { 'v1' : 'class' , 'v2' : 'message'}, inplace= True)

data.head()
data.info()
import matplotlib.pyplot as plt

count =pd.value_counts(data["class"], sort= True)

count.plot(kind= 'bar', color= ["blue", "orange"])

plt.title('Bar chart')

plt.legend(loc='best')

plt.show()
count.plot(kind = 'pie',autopct='%1.2f%%') # 1.2 is the decimal points for 2 places

plt.title('Pie chart')

plt.show()
data.groupby('class').describe()
data['length'] = data['message'].apply(len)

# swapping the columns

data = data[['message', 'length', 'class']]

data.head()
import re

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

def clean_message(message):

    message = re.sub("[^A-Za-z]", " ", message) #1

    message = message.lower() #2

    message = message.split() #3

    stemmer = PorterStemmer()   #4. to find the  root meaning word of each word         

    message = [stemmer.stem(word) for word in message if word not in set(stopwords.words("english"))] #5

    message = " ".join(message) #6 #Keeping cleaned words together

    return message
message = data.message[0]

print(message)
message = clean_message(message)

print(message)
messages = []

for i in range(0, len(data)):

    message = clean_message(data.message[i])

    messages.append(message)
data = data.drop(["message"],axis=1)

data['messages'] = messages

data.head()
#let's seperate the output and documents

y = data["class"].values

x = data["messages"].values
from sklearn.model_selection import train_test_split

#splitting the data in training and test set

xtrain , xtest , ytrain , ytest = train_test_split(x,y, test_size = 0.3, random_state = 1)

# test size is 0.3 which is 70 : 30

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english',max_df=0.5)



#fitting train data and then transforming it to count matrix#fitting 

x_train = vect.fit_transform(xtrain)

#print(x_train)



#transforming the test data into the count matrix initiated for train data

x_test = vect.transform(xtest)



# importing naive bayes algorithm

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()



#fitting the model into train data 

nb.fit(x_train,ytrain)



#predicting the model on train and test data

y_pred_test = nb.predict(x_test)

y_pred_train = nb.predict(x_train)



#checking accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,y_pred_test)*100)



#Making Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest,y_pred_test)

print(cm)
from sklearn.feature_extraction.text import CountVectorizer

vect1 = CountVectorizer(stop_words='english',max_df=0.5)



#fitting train data and then transforming it to count matrix#fitting 

x_train = vect1.fit_transform(xtrain)



#transforming the test data into the count matrix initiated for train data

x_test = vect1.transform(xtest)



# importing naive bayes algorithm

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()



#fitting the model into train data 

nb.fit(x_train,ytrain)



#predicting the model on train and test data

y_pred_test = nb.predict(x_test)

y_pred_train = nb.predict(x_train)



#checking accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,y_pred_test)*100)



#Making Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest,y_pred_test)

print(cm)
new_text = pd.Series('WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. valid 12 hours')

new_text_transform = vect.transform(new_text)

print(" The email is a" ,nb.predict(new_text_transform))
new_text = pd.Series(" Hello, how are you?")

new_text_transform = vect.transform(new_text)

print(" The email is a" ,nb.predict(new_text_transform))