# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv("../input/Book1.csv")
#checking the dataset

dataset.head(30) 
#Checking the dimensions of dataset....

dataset.shape
#Checking the null values in the dataset

dataset.isnull().sum()
dataset=dataset.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Time','Summary'],axis=1)
dataset.head()
#Counting all unique values of column "sentiment" from dataset 



dataset["Score"].value_counts()
#visualize the "Score" of each customer with the help of Bar plot 

dataset["Score"].value_counts().plot.bar(color='black')
dataset["Score"] = dataset["Score"].apply(lambda score: "1" if score > 3 else "0")
dataset.head()
dataset = dataset[['Text','Score']]

dataset.head()
#Bar plot for Score column

dataset["Score"].value_counts().plot.bar(color='Pink')
# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = [] #Creating an empty list

for i in range(0, 1000):

    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])#Remove all special character like('',:,;,%,*,@,$,!,#) 

    #Convert complete sentences into lowercase. 

    review = review.lower() 

    #Spliting each sentences into different words.

    review = review.split()

    #Stemming example is love-loving,loved,lovely.etc.

    ps = PorterStemmer()

    #Removing unnecessary words like is,of,for,this...and only focusing on main meaning of the sentence. 

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
print(corpus)
print(review)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 4199)

X= cv.fit_transform(corpus).toarray()

X = X.transpose()#Transposing here just to match the dimensionsns with y-target variable



y = dataset.iloc[:,-1].values
print(X)
print(y)
X.shape
y.shape
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
#confusion matrix.....

import seaborn as sns

plt.figure(figsize=(20,10))

plt.subplot(2,4,3)

plt.title("Multinomial naive bayes_cm")

sns.heatmap(cm,annot=True,cmap="prism",fmt="d",cbar=False)
#Check the accuracy

from sklearn.metrics import accuracy_score

print("The accuracy of naives bayes is:",accuracy_score(y_pred,y_test))