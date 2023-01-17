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

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string
df = pd.read_csv('../input/twitter-sample-dataset/twitter dataset.csv',encoding='iso-8859-1')

df
df.info()
df = df.drop(['gender_gold','profile_yn_gold','tweet_coord','tweet_id','tweet_location','user_timezone'],axis = 1)

df.head()
df.info()
df['gender'].unique()
df = df[df['gender'] != 'nan']

df.info()
df = df.dropna(subset = ['gender'])

df.info()
df = df.replace(np.nan, '', regex = True)

df.info()
df = df[df['gender'] != 'unknown']

df['gender'].unique()
df.shape
df = df.drop(['_unit_state', 'created', 'tweet_created', 'sidebar_color', 'link_color', 'profileimage', '_last_judgment_at', '_trusted_judgments'],axis = 1)

df.head()
df.shape
from sklearn.preprocessing import LabelEncoder

labelen = LabelEncoder()
df['gender1'] = labelen.fit_transform(df['gender'])

df['profile_yn1'] = labelen.fit_transform(df['profile_yn'])

df.head()
df['gender1'].value_counts()
df.describe()
df = df.sort_values('gender:confidence', ascending = True)

df
fig, ax = plt.subplots(figsize=(15,6))

ax.scatter(df['gender:confidence'], df['gender1'])

ax.set_xlabel('Gender Confidence')

ax.set_ylabel('gender')

plt.show()
df = df[df['gender:confidence']>=0.6]

df
df = df.sort_values('tweet_count', ascending = False)

df
df = df[df['tweet_count']>=1000]

df
df.describe()
import seaborn as sb
sb.boxplot(x=df['tweet_count'])
fig, ax = plt.subplots(figsize=(15,6))

ax.scatter(df['tweet_count'], df['gender1'])

ax.set_xlabel('Number of Tweets')

ax.set_ylabel('gender')

plt.show()
df = df[df['tweet_count']<=500000]

df
df['gender1'].value_counts()
male_df = df[df['gender1'] == 1][:4903]

female_df = df[df['gender1'] == 2]

female_df.shape,male_df.shape



df = male_df

df = df.append(female_df)
df.shape
df['gender1'].value_counts()
df.describe()
tweets = list(df['text'])

tweets[5]
def strip_all_entities(text):

    words = []

    entity_prefixes = ['@','#','\\']



    for word in text.split():

        word = word.strip()

        if word:

            if word[0] not in entity_prefixes:

                words.append(word)

    return ' '.join(words)





for i in range(0,len(tweets)):

    tweets[i] = tweets[i].lower()

    tweets[i] = strip_all_entities(tweets[i])



tweets[5]
def remove_links(text):

    words = []

    for word in text.split():

        if not 'https' in word:

            words.append(word)

    return ' '.join(words)





for i in range(0,len(tweets)):

    tweets[i] = remove_links(tweets[i])

    tweets[i] = tweets[i].replace("[^a-zA-Z#]"," ")

tweets[0:5]
def remove_punc(text):

    words = nltk.word_tokenize(text)

    words=[word for word in words if word.isalpha()]

    return ' '.join(words)



for i in range(0,len(tweets)):

    tweets[i] = remove_punc(tweets[i])



tweets[0:10]
stop_words = set(stopwords.words('english'))



def remove_stopwords(text):

    words = nltk.word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return ' '.join(words)



for i in range(0,len(tweets)):

    tweets[i] = remove_stopwords(tweets[i])



tweets[0:10]
from nltk.stem import WordNetLemmatizer



lemm = WordNetLemmatizer()



def get_root_words(text):

    words = nltk.word_tokenize(text)

    words = [lemm.lemmatize(word) for word in words]

    return " ".join(words)



for i in range(0,len(tweets)):

    tweets[i] = get_root_words(tweets[i])



tweets[0:10]
df['tweets'] = tweets



df.head()
df.info()
import seaborn as sb



plt.subplots(figsize=(20,15))

sb.heatmap(df.corr(), annot=True)
df.to_csv('twitter dataset_final.csv')

df
df.columns
df.info()
X = df[['gender:confidence','fav_number', 'retweet_count', 'tweet_count']].values

Y = df[['gender1']].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr.score(X_test, Y_test)
from sklearn.tree import DecisionTreeClassifier 
dtc=DecisionTreeClassifier(random_state = 0)
dtc.fit(X_train,Y_train)
dtc.score(X_test,Y_test)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train,Y_train)
gnb.score(X_test,Y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 0)

rf.fit(X_train,Y_train)
rf.score(X_test,Y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
knn.score(X_test,Y_test)
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(multi_class = 'multinomial', random_state = 0)

clf2 = RandomForestClassifier(n_estimators = 50, random_state = 0)

clf3 = DecisionTreeClassifier(random_state = 0)
vc = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dtc', clf3)], voting='hard')

vc.fit(X_train,Y_train)
vc.score(X_test,Y_test)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (1,3),max_features = 10000)



X = cv.fit_transform(df['tweets'])



df['gender'].unique()
y = df['gender1']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()

nb.fit(X_train,y_train)

pred1 = nb.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy_score(pred1,y_test)
dtc.fit(X_train,y_train)

pred2 = dtc.predict(X_test)



accuracy_score(pred2,y_test)
rf.fit(X_train,y_train)

pred3 = rf.predict(X_test)



accuracy_score(pred3,y_test)
X_test = X_test.toarray()

X_train = X_train.toarray()

lr.fit(X_train,y_train)

pred5 = lr.predict(X_test)



accuracy_score(pred5,y_test)
clf1 = LogisticRegression(multi_class='multinomial', random_state=0)

clf2 = RandomForestClassifier(n_estimators=50, random_state=0)

clf3 = MultinomialNB()



vc = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3)], voting='hard')

vc.fit(X_train,y_train)

pred6 = vc.predict(X_test)



accuracy_score(pred6,y_test)
from sklearn.feature_extraction.text import CountVectorizer



from yellowbrick.text import FreqDistVisualizer
male_df = df[df['gender'] == 'male']

female_df = df[df['gender'] == 'female']
cvm = CountVectorizer()

cvf = CountVectorizer()



Xm = cvm.fit_transform(male_df['tweets'])

Xf = cvf.fit_transform(female_df['tweets'])
featuresm   = cvm.get_feature_names()



visualizerm = FreqDistVisualizer(features=featuresm, orient='v',size=(1080, 720),n = 100)

visualizerm.fit(Xm)

visualizerm.show()
featuresf   = cvf.get_feature_names()



visualizerf = FreqDistVisualizer(features=featuresf, orient='v',size=(1080, 720),n = 100)

visualizerf.fit(Xf)

visualizerm.show()
df['fav_number']
import sympy as sp
lst=[]

for i in df['fav_number']:

    if sp.isprime(i)==True:

        lst.append('1')

    else:

        lst.append('0')
lst
arr=np.asarray(lst, dtype=np.int64)

arr
df['prime_numbers']=arr

df
df.info()
df=df[df['prime_numbers']==1]

df
df['gender'].value_counts()