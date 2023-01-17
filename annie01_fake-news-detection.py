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
real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
real.head()
fake.head()
real['label'] = 1

fake['label'] = 0

data = pd.concat([real, fake])
import seaborn as sns

sns.set_style("darkgrid")

sns.countplot(data['label']);
data.isnull().sum()
data.columns
import matplotlib.pyplot as plt

data['subject'].value_counts()

plt.figure(figsize = (10,10))

sns.set_style("darkgrid")

sns.countplot(data['subject']);
plt.figure(figsize = (10,10))

sns.set_style("dark")

chart = sns.countplot(x = "label", hue = "subject" , data = data , palette = 'muted')

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
data['text'] = data['title'] + " " + data['text']

data = data.drop(['title', 'subject', 'date'], axis=1)
from nltk.corpus import stopwords

from wordcloud import WordCloud



wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords.words('english'), 

                min_font_size = 10).generate(" ".join(data[data['label'] == 0].text)) 

  

# plot the word cloud for fake news data                      

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
from wordcloud import WordCloud



wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords.words('english'), 

                min_font_size = 10).generate(" ".join(data[data['label'] == 1].text)) 

  

# plot the WordCloud image for genuine news data                     

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
#splitting data for training and testing

import sklearn

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data['text'],data['label'],test_size=0.2, random_state = 1)
#Multinomial NB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import sklearn.metrics as metrics                                                 

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix





pipe = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', MultinomialNB())

])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)



score = metrics.accuracy_score(y_test, prediction)

print("accuracy:   %0.3f" % (score*100))

cm = metrics.confusion_matrix(y_test, prediction, labels=[0,1])







fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, prediction),

                                show_absolute=True,

                                show_normed=True,

                                colorbar=True)

plt.show()



#SVM

from sklearn.svm import LinearSVC

pipe = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', LinearSVC())

])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)



score = metrics.accuracy_score(y_test, prediction)

print("accuracy:   %0.3f" % (score*100))

cm = metrics.confusion_matrix(y_test, prediction, labels=[0,1])







fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, prediction),

                                show_absolute=True,

                                show_normed=True,

                                colorbar=True)

plt.show()

#Passive Aggressive Classifier

from sklearn.linear_model import PassiveAggressiveClassifier

pipe = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf',  PassiveAggressiveClassifier())

])



model = pipe.fit(x_train, y_train)

prediction = model.predict(x_test)



score = metrics.accuracy_score(y_test, prediction)

print("accuracy:   %0.3f" % (score*100))

cm = metrics.confusion_matrix(y_test, prediction, labels=[0,1])







fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, prediction),

                                show_absolute=True,

                                show_normed=True,

                                colorbar=True)

plt.show()