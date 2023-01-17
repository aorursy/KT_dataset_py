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
import pandas as pd

Hotel_Reviews = pd.read_csv("../input/dadada.csv")
Hotel_Reviews.info()
Hotel_Reviews.head()
import re #Importing Regex
#review =re.sub('[^a-zA-Z]', ' ', Hotel_Reviews['Negative_Review'][0])
#review
#review=review.lower()
#review
import nltk #This Library is used for Sentiment Analysis
#Removing Stop words

from nltk.corpus import stopwords

#nltk.download('stopwords'), doesn't work since we are working on cloud
#review = review.split()
#review=[word for word in review if not word in stopwords.words('english')]

#Run for once to get an understanding
#review
#Stemming

#from nltk.stem.porter import PorterStemmer #DONOT USE
#ps=PorterStemmer() #DONOTUSE
#review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
#review =' '.join(review)
#review
corpus=[] #empty list creation

#create the loop

for i in range(0, 48979):

    review =re.sub('[^a-zA-Z]', ' ', Hotel_Reviews['Negative_Review'][i]) #change it to i

    review=review.lower()

    review = review.split()

    review =' '.join(review)

    corpus.append(review)
#Creating the BOW model through the process of tokenization, we can use count vectorizor 

#(like marking 1 for the all the 1's, very hard, hence we can use Count Vectorizor)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

#Creation of Sparse Matrix

X=cv.fit_transform(corpus).toarray()
#X  #use this for lower ranges
X.itemsize
y= Hotel_Reviews.iloc[:, 3].values.astype(int)
X.shape
y.shape
#Naive Bayes classification

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
# Accuracy

(362+1080+1725)/48979
import wordcloud

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 
comment_words = ' '

stopwords = set(STOPWORDS) 
Words=Hotel_Reviews.iloc[:, 6]
# iterate through the csv file 

for val in words: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
Words_Pos=Hotel_Reviews.iloc[:, 9]
# iterate through the csv file 

for val in Words_Pos: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  
# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 