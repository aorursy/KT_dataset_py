# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/restaurantreviews/Restaurant_Reviews.tsv', delimiter='\t', quoting = 3) 

#3 means ignoring double quotes 
dataset.head()
import re 
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
corpus=[]

#Removing puncutations, and number

#^ represents don't want to remove

#Keeping the letters a to z and A to Z with space

#collection of text is called as corpus

#Creating the Bag of words model

corpus = []

for i in range(0,1000):

    

    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])

    

    #putting all letters in lower case

    review = review.lower()

    

    #Removing stopwords from string and stemming the word

    

    #Stemming is used for make the words to normal form (root) like loved will become like love, loving will become love and capital letter of first letter will become small

    #Stemming is taking of root of the word

    

    review = review.split()

    

    ps = PorterStemmer()

    

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    

    #Joining the words to make a string

    

    review =' '.join(review)

    corpus.append(review)
corpus[1:5]
#tokenizer
from sklearn.feature_extraction.text import CountVectorizer

#max_features is used to remove non relavent words

cv = CountVectorizer(max_features = 1500)

#Spars metrics in NLP

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values
#Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X[0]
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
cm