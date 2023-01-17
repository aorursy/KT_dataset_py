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
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
# Importing the dataset
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df
df['length'] = df['review'].apply(len)
df.head()
df.sentiment = (df.sentiment.replace({'positive': 1, 'negative': 0})).values
df['length'].plot(bins=100,kind='hist') 
df.length.describe()
#Longest review
df[df['length'] == 13704]['review'].iloc[0]
#Shortest review
df[df['length'] == 32]['review'].iloc[0]
df.hist(column='length', by='sentiment',figsize=(12,4))

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


ps = PorterStemmer()
lem = WordNetLemmatizer()
corpus = []

for i in range(df.shape[0]):
    soup = BeautifulSoup(df['review'][i], "html.parser")
    review = soup.get_text()
    review = re.sub('[^a-zA-Z]', ' ', df['review'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    if (i%1000 == 0):
        print(i)
# Creating the Bag of Words model


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(ngram_range=(1,3), max_features=10000)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

print(x.shape)
print(y.shape)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Naive Bayes
#Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# Classification report
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, labels=None)
print(report)
wordcloud = WordCloud(width = 800, height = 800,background_color ='white',min_font_size = 10).generate(" ".join(corpus)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
