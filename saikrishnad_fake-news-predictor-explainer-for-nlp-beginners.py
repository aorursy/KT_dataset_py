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
df = pd.read_csv('/kaggle/input/fake-news/train.csv')

df.head()
Y = df.label

Y.tail()
X = df.drop(['label'],axis=1)

X.tail()
df.groupby('author').sum().sort_values(by=['label'],ascending=False).head(10)
authors_df = pd.DataFrame()

authors_df = df.groupby('author').count().reset_index()[['author','label']]

authors_df.columns = ['Author','Total number of articles written']

authors_df
authors_df['Number of fake articles written'] = df.groupby('author').sum().reset_index()['label']

authors_df['Percentage Fake'] = 100*authors_df['Number of fake articles written']/authors_df['Total number of articles written']

authors_df.sort_values(by=['Total number of articles written'],ascending=False)
fil = (authors_df['Total number of articles written'] >= 10)

imp_authors = authors_df[fil].reset_index().drop('index',axis=1)

imp_authors
fil2 = (imp_authors['Percentage Fake'] >= 1)

fake_authors = imp_authors[fil2].reset_index().drop('index',axis=1)

fake_authors.sort_values(by=['Total number of articles written'],ascending=False)
fake_authors['Percentage Fake'].value_counts()
fil3 = (imp_authors['Percentage Fake'] < 1)

credible_authors = imp_authors[fil3].reset_index().drop('index',axis=1)

credible_authors.sort_values(by=['Total number of articles written'],ascending=False)
credible_authors['Percentage Fake'].value_counts()
#Removing stopwords

from nltk.corpus import stopwords

#These are the following stopwords that will have to be removed

set(stopwords.words('english'))
example_words = ['if','Walking','can','Bathed','Consultant']

key_words = [word for word in example_words if not word in stopwords.words('english')]

key_words
#Stemming

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()



example_words = ['walking','reading','consultant','Laughed']

stemmed_words = [ps.stem(word) for word in example_words]

stemmed_words
# Removing Special Characters:

import regex as re

s = "This is an explainer on NLP #1 for $10"

without_specialchar = re.sub('[^a-z,A-Z,0-9]',' ',s)

without_specialchar
without_specialchar.lower().split()
df.isnull().sum()
articles = df.dropna().reset_index().drop('index',axis=1)

articles.tail()
corpus = []

for i in range(0,len(articles)):

    title = re.sub('[^a-z,A-Z]',' ',articles['title'][i]).lower().split()

    #Removing stopwords and stemming

    title_keywords = [ps.stem(word) for word in title if not word in stopwords.words('english')]

    title_processed = ' '.join(title_keywords)

    corpus.append(title_processed)
corpus
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
text = ['This is an NLP explainer on NLP #1 for $10 NLP']



#Fit each unique word in the string to a unique number. 

vectorizer.fit(text)
print(vectorizer.vocabulary_)
# Create a vector with 'count values' of each unique word. 

vector = vectorizer.transform(text)

vector
print(vector.shape)

print(type(vector))

print(vector.toarray())
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))

#Fit the words to unique number and transform the text into a vector. 

vectors = cv.fit_transform(corpus).toarray()

vectors
print(len(cv.vocabulary_))
cv.vocabulary_
vectors_df = pd.DataFrame(vectors, columns = cv.get_feature_names())

vectors_df.head()
# Prepare the input and output variables

X = vectors

y = articles['label']
# Split the data into test and train datasets. 

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

from sklearn import metrics

import itertools





score = metrics.accuracy_score(y_test, y_pred)

print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

logreg.get_params()