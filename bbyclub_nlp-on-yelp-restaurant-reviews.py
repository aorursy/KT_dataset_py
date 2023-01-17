# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # Import Regular Expression Library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Restaurant_Reviews.csv")
#Exchange the characters not between in "a-z" and "A-Z" with space character

comment=re.sub('[^a-zA-Z]',' ',data['Review'][0])

#Now data is not DataFrame anymore. It's string!
comment=comment.lower() #transfom all the characters into lower case
comment=comment.split() #transform the sentence into word list
#remove the stopwords

from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')

print(stopwords_en)
#Stemming and Lemmatization

from nltk.stem.porter import PorterStemmer

ps= PorterStemmer() 

comment=[ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]

#If the word is not stopwords, throw it into the list

#Since we write in square brackets, the values returned from the function will be defined as a list

comment= ' '.join(comment) #Merge all words in comment with a space between them and put them in comment. Comment is string now
#repeat all the steps for all the reviews in Dataset

comments=[]

for i in range(1000):

    comment=re.sub('[^a-zA-Z]',' ',data['Review'][i])

    comment=comment.lower() #transfom all the characters into lower case

    comment=comment.split()

    comment=[ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]

    comment= ' '.join(comment) #Merge all words in comment with a space between them and put them in comment. Comment is string now

    comments.append(comment)

comments
from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(max_features=1000) #Take 2000 words most common used

X = cv.fit_transform(comments).toarray()#independent variable

y = data.iloc[:,1].values  #dependent variable
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

gnb= GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,y_pred)

print(cm)