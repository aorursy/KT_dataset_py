import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/amazon-e-commerce-data-set/ecommerceDataset.csv",names=['Type','Text'])
df.head()
df['Type'].value_counts()
df.shape
df.Text.nunique()
df.groupby('Type').describe()
df.columns
sns.heatmap(df.isnull())
df.dropna(axis=0,inplace=True)
import nltk

import string
from nltk.corpus import stopwords
string.punctuation
df['Text'][10] #Original text
kk="".join([r for r in df['Text'][10] if r not in string.punctuation]) #depunctuated text
kk
stopwords.words("english")
kk.split(" ") #orginal text
[w for w in kk.split(" ") if w not in stopwords.words("english")] # after removing stopwords
def fun(w):

    w=[r for r in w if r not in string.punctuation] #removes punctuation

    w="".join(w)

    return [ x for x in w.split(" ") if x.lower() not in stopwords.words("english")]
from sklearn.feature_extraction.text import CountVectorizer 
vec=CountVectorizer(analyzer=fun).fit(df['Text'])
vec=vec.transform(df['Text'])
from sklearn.feature_extraction.text import TfidfTransformer

vec_tfidf=TfidfTransformer().fit(vec)

vec_tfidf=vec_tfidf.transform(vec)
from sklearn.model_selection import train_test_split
X=df['Text']

y=df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

r=RandomForestClassifier(n_estimators=100)
from sklearn.pipeline import Pipeline

pp=Pipeline([

    ('b',CountVectorizer()),

    ('C',TfidfTransformer()),

    ('r',RandomForestClassifier(n_estimators=100))

])
pp.fit(X_train,y_train)
from sklearn.metrics import classification_report
print(classification_report(pp.predict(X_test),y_test))
df.loc[101]['Type'] #Actual Value
pp.predict(['AsianHobbyCrafts Wooden Embroidery Hoop Ring Frame (3 Pieces) Style Name:Assorted A   Asian Hobby Crafts embroidery collection comprises of embroidery frames (in various sizes), cross stitch fabric, embroidery tools, embroidery wool. This embroidery hoop frame is made of well finished wood with a easy-to-adjust screw mounted on the frame to tighten the fabric. Cross stitch art is a phenomenal art form which involves intricate stitching techniques to form beautiful designs on fabric.'])[0]

#Original value