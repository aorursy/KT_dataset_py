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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import os

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer
messages = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

messages.head()
messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
messages.rename(columns={'v1':'Label','v2':'Message'},inplace = True)

messages.info()
sns.countplot(messages.Label,palette="twilight")

plt.xlabel('Label')

plt.title('Number of ham and spam messages')
ax = sns.countplot(y="Label", data=messages,palette="twilight")

plt.xlabel('Count')

plt.title('Number of ham and spam messages')



total = len(messages['Label'])

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))



plt.show()
ps = PorterStemmer # You can use stemming but here I am using Lemmatization.

lmr = WordNetLemmatizer

corpus = []

for i in range( 0, len(messages)):

    review = re.sub('[^a-zA-Z]', ' ', messages['Message'][i]) # Cleaning all special characters and numbers, keeping words only

    review = review.lower() # Making all the words to lower case.

    review = review.split()

    

    review = [lmr.lemmatize('word',word) for word in review if word not in set(stopwords.words('english'))]

    # This is list compreshession adding the words which are not avaibale in stopwords.

    review = ' '.join(review)

    corpus.append(review) # adding all words to Corpus
# Creating a Bag Of Words model:

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000) # Taking the 5000 important words.

X = cv.fit_transform(corpus).toarray() # Fitting the model to Corpus and converting it to array.
Y = pd.get_dummies(messages['Label'])

# No need to specify teo catregorical column here we can just define one culumn (ie:if 0-> Ham or if ->1 Spam):

Y = Y.iloc[:,1].values
X.shape
Y.shape
# Spliting the Data into train and test:

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size= 0.2,random_state = 5)
# Training the model with Naive Bayes Classifier:

from sklearn.naive_bayes import MultinomialNB

spam_detection_model=MultinomialNB().fit(X_train,Y_train)
# The predected output of our model

Y_pred = spam_detection_model.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(Y_test,Y_pred) #It shows the total Right Predictions (960+140) and Wrong Predictions(10+5).

print(confusion_m)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred) # It shows the accuracy of the model.

print (accuracy)