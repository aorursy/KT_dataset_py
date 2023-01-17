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
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
data= pd.read_csv('../input/restaurant-reviews/Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
data.head()
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus= []

for i in range(len(data)):

    review=re.sub('[^a-zA-Z]',' ',data['Review'][i]) #not a-z and A-Z only non letter will be replace by space

    review=review.lower()

    review=review.split()

    ps= PorterStemmer()

    all_stopwords= stopwords.words('english')

    all_stopwords.remove('not') #it will remove not word from the stopword

    review= [ps.stem(word) for word in review if not word in set(all_stopwords)]

    review= ' '.join(review)

    corpus.append(review)

corpus
from wordcloud import WordCloud

from os import path



wordcloud= WordCloud(

    

    background_color="white",

    width=3400,

    height= 1200

    ).generate(" ".join(corpus))

plt.figure(figsize=(40,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer()



X= cv.fit_transform(corpus).toarray()

Y=data.iloc[:,-1].values
len(X[0])


#split the data



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.22, random_state = 0)


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print("Accuracy score {}".format(acc_logreg))

cm=confusion_matrix(y_test,y_pred)

print("Confusion metrics  {}".format(cm))

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

print("Confusion metrics  {}".format(cm))

print(accuracy_score(y_test,y_pred))
new_review = 'I love this restaurant so much'

new_review = re.sub('[^a-zA-Z]', ' ', new_review)

new_review = new_review.lower()

new_review = new_review.split()

ps = PorterStemmer()

all_stopwords = stopwords.words('english')

all_stopwords.remove('not')

new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]

new_review = ' '.join(new_review)

new_corpus = [new_review]

new_X_test = cv.transform(new_corpus).toarray()

new_y_pred = gaussian.predict(new_X_test)

print(new_y_pred)