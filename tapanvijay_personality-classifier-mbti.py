

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

from bs4 import BeautifulSoup

from sklearn.model_selection import cross_validate

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

mbti_df = pd.read_csv('../input/mbti_1.csv')
# Labels that need to be removed from posts

lbl_rmv=list(mbti_df['type'].unique())

lbl_rmv = [item.lower() for item in lbl_rmv]
from nltk.corpus import stopwords

stop = stopwords.words('english')



from nltk.stem.porter import PorterStemmer
for i in range(0,8675) :  

    mbti_df['posts'][i] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', mbti_df['posts'][i])

    mbti_df['posts'][i] = re.sub("[^a-zA-Z]", " ", mbti_df['posts'][i])

    mbti_df['posts'][i] = re.sub(' +', ' ', mbti_df['posts'][i]).lower()

    for j in range(0,16):

        mbti_df['posts'][i]=re.sub(lbl_rmv[j], ' ', mbti_df['posts'][i])

        

mbti_df['posts'] = mbti_df['posts'].str.strip()



#corpus creation and stopwords and porterstemming 

def pre_process(post):

    posts = re.sub('\s+', ' ', post)

    posts = posts.lower()

    posts = posts.split()

    posts = [word for word in posts if not word in set(stopwords.words('english'))]

    ps = PorterStemmer()

    posts = [ps.stem(word) for word in posts]

    posts = ' '.join(posts)

    return posts

    

corpus = mbti_df["posts"].apply(pre_process)
# converting the personality types to 8 respective binary identifiers(I-E,N-S,T-F,J-P)

map1 = {"I": 0, "E": 1}

map2 = {"N": 0, "S": 1}

map3 = {"T": 0, "F": 1}

map4 = {"J": 0, "P": 1}

mbti_df['I-E'] = mbti_df['type'].astype(str).str[0]

mbti_df['I-E'] = mbti_df['I-E'].map(map1)

mbti_df['N-S'] = mbti_df['type'].astype(str).str[1]

mbti_df['N-S'] = mbti_df['N-S'].map(map2)

mbti_df['T-F'] = mbti_df['type'].astype(str).str[2]

mbti_df['T-F'] = mbti_df['T-F'].map(map3)

mbti_df['J-P'] = mbti_df['type'].astype(str).str[3]

mbti_df['J-P'] = mbti_df['J-P'].map(map4)
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)

features = cv.fit_transform(mbti_df['posts']).toarray()

IE= mbti_df.iloc[:, 2].values

NS= mbti_df.iloc[:, 3].values

TF=mbti_df.iloc[:, 4].values

JP=mbti_df.iloc[:, 5].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

features_train, features_test, IE_train, IE_test, NS_train, NS_test, TF_train, TF_test, JP_train, JP_test = train_test_split(features, IE,NS,TF,JP, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier



# fit model on training data

IEB = XGBClassifier()

IEB.fit(features_train, IE_train)

ieb_train=IEB.score(features_train,IE_train)

ieb_test=IEB.score(features_test,IE_test)



NSB = XGBClassifier()

NSB.fit(features_train, NS_train)

nsb_train=NSB.score(features_train,NS_train)

nsb_test=NSB.score(features_test,NS_test)





TFB = XGBClassifier()

TFB.fit(features_train, TF_train)

tfb_train=TFB.score(features_train,TF_train)

tfb_test=TFB.score(features_test,TF_test)



JPB = XGBClassifier()

JPB.fit(features_train, JP_train)

jpb_train=JPB.score(features_train,JP_train)

jpb_test=JPB.score(features_test,JP_test)
print('Label I-E train score is :',ieb_train)

print('Label I-E test score is :',ieb_test)

print('Label N-S train score is :',nsb_train)

print('Label N-S test score is :',nsb_test)

print('Label T-F train score is :',tfb_train)

print('Label T-F test score is :',tfb_test)

print('Label J-P train score is :',jpb_train)

print('Label J-P test score is :',jpb_test)