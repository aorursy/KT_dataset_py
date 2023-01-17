# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#After uploading data set we print whole data set with the help of print command 
import pandas as pd
mail = pd.read_csv("../input/mail-checker/spam_or_not_spam.csv",encoding="ISO-8859-1")
print(mail)
#an display information such as the number of rows and columns, the total memory usage, the data type of each column, and the number of non-NaN elements.
mail.info()
#it gives  the length of any data sheet i.e no of rows we can say 
print(len(mail))
#it gives the numbers of columns present in data sheet 
print("columns:")
print(len(mail.columns))


#it gives the numbers of rows and columns present in data sheet
row,col=mail.shape
print("rows:",row)
print("columns:",col)

# Print the shape of the dataset to get the no. of rows and columns.
print(mail.shape)
print(mail.head())
print(mail.columns)
print(mail.describe)
# look at the first few rows
mail.shape
mail.head(10)
 #count the number of duplicated row in our dataset
#duplicated(ign_mail) %>%
    # a+b = c
    # sum (a,b)
mail.drop_duplicates(inplace=True)

print(mail.drop_duplicates)
#show the number of missing data in each column (NAN, NaN,na ) etc
print(mail.isnull().sum())
# Shape of training data (row,col)
print(mail.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (mail.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# Drop the row having missing value
mail.dropna(inplace = True)
print(mail.shape)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string
def process_text (email):
    #remove punctuation
  #remove stopwords 
  #return a list of clean text words

  nopunc = [char for char in email if char in string.punctuation]
  nopunc = ''.join(nopunc)

  clean_words= [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return clean_words
print(mail['email'].head().apply(process_text))

#convert a collection of text into matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
message_bow= CountVectorizer(analyzer= process_text).fit_transform(mail['email'])
#split data into 70% training and 30% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(message_bow, mail['label'], test_size=0.30, random_state=0)
print(message_bow.shape)
#create and train the Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)
#print the predictions
#print(classifier.predict(X_train))

#print the actual values
print(y_train.values)
#evaluate the model on the training dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred= classifier.predict(X_train)
print(classification_report(y_train,pred))
print()

print('Confusion matrix : \n', confusion_matrix (y_train,pred))
print()
print('Accuracy : ', accuracy_score(y_train, pred))

#print the predictions

print(classifier.predict(X_test))

#print the actual values
print(y_test.values)

#evaluate the model on the test dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred= classifier.predict(X_test)
print(classification_report(y_test,pred))
print()

print('Confusion matrix : \n', confusion_matrix (y_test,pred))
print()
print('Accuracy : ', accuracy_score(y_test,pred))