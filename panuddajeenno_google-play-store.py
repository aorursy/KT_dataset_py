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
# data analysis and wrangling

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#input ข้อมูลมาเก็บไว้ในตัวแปร

df = pd.read_csv('../input/googleplaystore/googleplaystore.csv')

reviews = pd.read_csv('../input/ggstore/googleplaystore_user_reviews.csv')

df.head()
df.tail()
print(df.columns.values)
df.info()
print(df.dtypes)
df.rename(columns={'App':'app','Category':'category','Rating':'rating','Reviews':'reviews','Size.':'size','Installs.':'installs', 'Type':'type','Price':'price','Content Rating':'content_rating','Genres':'genres','Last Updated':'last_updated','Current Ver':'current_ver','Android Ver':'android_ver'},inplace=True)

df.head()
df['category'].unique()
#คำนวณจำนวนแอปของแต่ละประเภท

print(type(df['category']))

popular=df.groupby('category').size().unique

print(popular)

genre_list=df['category'].values.tolist()
df.isnull().any()
pd.set_option('precision', 3)

df.describe()
sns.heatmap(df.corr(), annot=True)
df.groupby('category')['rating'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Category', fontsize = 20)

plt.ylabel('Count of rating', fontsize = 20)

plt.title('Category vs Count of rating', fontsize = 30)
df.rating.hist();

plt.xlabel('Rating')

plt.ylabel('Frequency')
df.category.unique()
df.category.value_counts().plot(kind='bar')
df.content_rating.unique()
df.content_rating.value_counts().plot(kind='bar')

plt.yscale('log')
df.last_updated.head()
from datetime import datetime,date

temp=pd.to_datetime(df.last_updated)

temp.head()
df['Last_Updated_Days'] = temp.apply(lambda x:date.today()-datetime.date(x))

df.Last_Updated_Days.head()
g = sns.catplot(x="category",y="rating",data=df, kind="box", height = 10 ,

palette = "Set1")

g.despine(left=True)

g.set_xticklabels(rotation=90)

g.set( xticks=range(0,34))

g = g.set_ylabels("rating")

plt.title('Boxplot of Rating VS Category',size = 20)
df['Installs'].unique()
Sorted_value = sorted(list(df['Installs'].unique()))

df['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
size=[8895,753]

sentiment = ['Free', 'Paid']

colors = ['yellow', 'red']

plt.pie(size, labels=sentiment, colors=colors, startangle=180, autopct='%.1f%%')

plt.title('% Free vs Paid Apps')

plt.show()
paided = df[df['type'] == 'Paid']
apppaid = paided['category'].value_counts()

apppaid = apppaid.reset_index()

apppaid = apppaid[:10]

plt.figure(figsize=(10,5))

plt.pie(x = list(apppaid['category']), labels=list(apppaid['index']), autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)

plt.title('% Distribution of Paided Apps Categories')
reviews.dropna(inplace=True)

reviews.isnull().sum()
reviews['Translated_Review'] = reviews.Translated_Review.str.replace("[^a-zA-Z#]", " ")
reviews.head()
reviews = reviews.reset_index().drop('index',axis=1)
reviews['Translated_Review'].apply(lambda x: '')
reviews.Sentiment.value_counts()
sns.countplot(reviews.Sentiment)
from scipy.stats import stats

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
x = reviews.Translated_Review

y = reviews.Sentiment
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.3)
# Logistic

review_lr = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', LogisticRegression())])



# Naïve Bayes:

review_nb = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', MultinomialNB())])



# Linear SVC:

review_svc = Pipeline([('tfidf', TfidfVectorizer()),

                     ('review', LinearSVC())])
def model(obj,name):

    ypred = obj.fit(xtrain,ytrain).predict(xtest)

    return print(name,"\n\n",

                "Accuracy Score:- ",accuracy_score(ytest,ypred),"\n\n Confusion Matrix:- \n",confusion_matrix(ytest,ypred),

                "\n\n Classification Report:- \n",classification_report(ytest,ypred))
model(review_lr,"Logistic Regression")
model(review_nb,"Naive Bayes")
model(review_svc,"Support Vector Classifier")