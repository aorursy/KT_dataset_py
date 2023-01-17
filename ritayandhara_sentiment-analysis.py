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
df=pd.read_csv('/kaggle/input/tweet-sentiments/tweet_sentiments.csv')
df['sentiment'].value_counts()
df.head()
# One review

df['tweet'][0]
#df=df.sample(100)
df.shape
df.info()
df['sentiment'].replace({'positive':1,'negative':-1, 'neutral':0},inplace=True)
df.head()
import re

clean = re.compile('<.*?>')

re.sub(clean, '', df.iloc[2].tweet)
# Function to clean html tags

def clean_html(text):

    clean = re.compile('<.*?>')

    return re.sub(clean, '', text)
df['tweet']=df['tweet'].apply(clean_html)
df.sample(3)
# converting everything to lower



def convert_lower(text):

    return text.lower()
df['tweet']=df['tweet'].apply(convert_lower)
# function to remove special characters



def remove_special(text):

    x=''

    

    for i in text:

        if i.isalnum():

            x=x+i

        else:

            x=x + ' '

    return x
remove_special(' th%e @ classic use of the word.it is called oz as that is the nickname given to the oswald maximum security state penitentary. it focuses mainly on emerald city, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. em city is home to many..aryans, muslims, gangstas, latinos, christians, italians, irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.i would say the main appeal of the show is due to the fact that it goes where other shows wouldnt dare. forget pretty pictures painted for mainstream audiences, f')
df['tweet']=df['tweet'].apply(remove_special)
# Remove the stop words

import nltk
from nltk.corpus import stopwords
stopwords.words('english')
df


def remove_stopwords(text):

    x=[]

    for i in text.split():

        

        if i not in stopwords.words('english'):

            x.append(i)

    y=x[:]

    x.clear()

    return y
df['tweet']=df['tweet'].apply(remove_stopwords)
df
# Perform stemming



from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
y=[]

def stem_words(text):

    for i in text:

        y.append(ps.stem(i))

    z=y[:]

    y.clear()

    return z

        
stem_words(['I','loved','loving','it'])
df['tweet']=df['tweet'].apply(stem_words)
df
# Join back



def join_back(list_input):

    return " ".join(list_input)

    
df['tweet']=df['tweet'].apply(join_back)
df['tweet']
df
X=df.iloc[:,1:2].values
X
X.shape
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=500)
X=cv.fit_transform(df['tweet']).toarray()
X
X.shape
X[0].mean()
y=df.iloc[:,-1].values
y.shape
# X,y

# Training set

# Test Set(Already know the result)
from sklearn.model_selection import train_test_split



X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
clf1=GaussianNB()

clf2=MultinomialNB()

clf3=BernoulliNB()
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
y_pred1=clf1.predict(X_test)

y_pred2=clf2.predict(X_test)

y_pred3=clf3.predict(X_test)
y_test.shape
y_pred1.shape
from sklearn.metrics import accuracy_score
print("Gaussian :",accuracy_score(y_test,y_pred1))

print("Multinomial :",accuracy_score(y_test,y_pred2))

print("Bernaulli :",accuracy_score(y_test,y_pred3))
import pickle
pickle.dump(X,open('X.pkl','wb'))

pickle.dump(y,open('y.pkl','wb'))
X.shape
y.shape