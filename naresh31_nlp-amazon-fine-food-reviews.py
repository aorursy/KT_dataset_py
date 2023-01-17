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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/Reviews.csv")
df=df.sample(1000)
df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)
df.shape
df.head()
df['Score'].value_counts()
df['Score'].isna().sum()
x=[]

for i in df['Score'] :

    if i<3 :

        x.append(0)

    else :

        x.append(1)
df['target']=x
df_new=df[['Text','Summary','Score','target']]
df_new.head()
special_char=[]

for i in df_new['Text']:

    for j in i:

        if j.isalpha()==False:

            special_char.append(j)
alpha_char=[]

num_char=[]

for i in df_new['Text'].str.split():

    for j in i:

        if j.isalpha()==False:

            num_char.append(j)

        else:

            alpha_char.append(j)
len(alpha_char)
len(num_char)
from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
x=[]

for i in alpha_char:

    for j in stop:

        if i not in j:

            x.append(i)
x = set(x)
len(x)
import nltk

porter = nltk.PorterStemmer()
words_stemming=[]

for i in x:

       words_stemming.append(porter.stem(i))
words_stemming[0:10]
import nltk

nltk.download('wordnet')
WNlemma = nltk.WordNetLemmatizer()

word_list_lem=[WNlemma.lemmatize(t) for t in words_stemming]
words_lem=[]

for i in words_stemming:

    WNlemma = nltk.WordNetLemmatizer()

    words_lem.append(WNlemma.lemmatize(i))
len(words_lem)
print(len(set(words_stemming)))

print(len(set(words_lem)))
import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt
x=" "

for i in words_lem:

       x=x+" "+i
x=str(x)

x=x[0:10000]
wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=2500,

                      height=2000

                     ).generate(x)

plt.figure(1,figsize=(13, 13))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df_new.shape
len(words_lem)
df_new.columns
df_new.values
x=[]

for j in words_lem:

    for i in df_new.Text:

        if j in i:

            x.append(1)

        else:

            x.append(0)
import numpy as np

x=np.array(x)
len(x)
df_lem=pd.DataFrame(x.reshape(1000,6787))
df_lem.columns=words_lem
df_lem.head()
y=df_new['target']

x=df_new['Text']
from sklearn.model_selection import train_test_split

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer



# Fit the CountVectorizer to the training data

vect = CountVectorizer().fit(X_train)

# transform the documents in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)

X_test_vectorized = vect.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=2)

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier()

model.fit(X_train_vectorized, y_train)

model.predict(X_test_vectorized)

from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(model.predict(X_test_vectorized),y_test))

print(accuracy_score(model.predict(X_test_vectorized),y_test))