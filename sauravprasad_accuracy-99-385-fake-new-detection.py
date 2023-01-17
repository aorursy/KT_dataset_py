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
#loading the data

fake=pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")

true=pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
#Creating a category for whether fake or not

#where 1 stand for fake news and 0 stands for true news



fake["category"]=1

true["category"]=0
#joining the data the two data frame and reseting index

df=pd.concat([fake,true]).reset_index(drop=True)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#creating a count plot for category column

fig = plt.figure(figsize=(10,5))







graph = sns.countplot(x="category", data=df)

plt.title("Count of Fake and True News")



#removing boundary

graph.spines["right"].set_visible(False)

graph.spines["top"].set_visible(False)

graph.spines["left"].set_visible(False)



#annoting bars with the counts  

for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.2,height ,ha="center",fontsize=12)
#creating a count plot for subject column

fig = plt.figure(figsize=(10,5))







graph = sns.countplot(x="subject", data=df)

plt.title("Count of Subjecs")



#removing boundary

graph.spines["right"].set_visible(False)

graph.spines["top"].set_visible(False)

graph.spines["left"].set_visible(False)



#annoting bars with the counts  

for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.2,height ,ha="center",fontsize=12)
#checking the missing values in each columns

df.isna().sum()*100/len(df)
#checking if there is empty string in TEXT column

blanks=[]



#index,label and review of the doc

for index,text in df["text"].iteritems(): # it will iter through index,label and review

    if text.isspace(): # if there is a space

        blanks.append(index) #it will be noted down in empty list



len(blanks)
#instead of dropping these values we are going to merge title with text



df["text"] =df["title"]+df["text"]



#we only need two columns rest can be ignored



df=df[["text","category"]]
#importing libraries for cleaning puprose



from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

import spacy

import re

#loading spacy library

nlp=spacy.load("en_core_web_sm")



#creating instance

lemma=WordNetLemmatizer()
#creating list of stopwords containing stopwords from spacy and nltk



#stopwords of spacy

list1=nlp.Defaults.stop_words

print(len(list1))



#stopwords of NLTK

list2=stopwords.words('english')

print(len(list2))



#combining the stopword list

Stopwords=set((set(list1)|set(list2)))

print(len(Stopwords))
#text cleaning function

def clean_text(text):

    

    """

    It takes text as an input and clean it by applying several methods

    

    """

    

    string = ""

    

    #lower casing

    text=text.lower()

    

    #simplifying text

    text=re.sub(r"i'm","i am",text)

    text=re.sub(r"he's","he is",text)

    text=re.sub(r"she's","she is",text)

    text=re.sub(r"that's","that is",text)

    text=re.sub(r"what's","what is",text)

    text=re.sub(r"where's","where is",text)

    text=re.sub(r"\'ll"," will",text)

    text=re.sub(r"\'ve"," have",text)

    text=re.sub(r"\'re"," are",text)

    text=re.sub(r"\'d"," would",text)

    text=re.sub(r"won't","will not",text)

    text=re.sub(r"can't","cannot",text)

    

    #removing any special character

    text=re.sub(r"[-()\"#!@$%^&*{}?.,:]"," ",text)

    text=re.sub(r"\s+"," ",text)

    text=re.sub('[^A-Za-z0-9]+',' ', text)

    

    for word in text.split():

        if word not in Stopwords:

            string+=lemma.lemmatize(word)+" "

    

    return string

#cleaning the whole data

df["text"]=df["text"].apply(clean_text)
from wordcloud import WordCloud
#True News

plt.figure(figsize = (20,20))

Wc = WordCloud(max_words = 500 , width = 1600 , height = 800).generate(" ".join(df[df.category == 0].text))

plt.axis("off")

plt.imshow(Wc , interpolation = 'bilinear')
#creating more intiuive wordcloud 



#pil is pillow and used for image manupulation

from PIL import Image

#creating a mask of thumb

thumb="../input/images-coud/thumbs-up.png"

icon=Image.open(thumb)

mask=Image.new(mode="RGB",size=icon.size, color=(255,255,255))

mask.paste(icon, box=icon)



rgb_array=np.array(mask)
#True News

plt.figure(figsize = (10,10))

Wc = WordCloud(mask=rgb_array,max_words = 2000 , width = 1600 ,

               height = 800)



Wc.generate(" ".join(df[df.category == 0].text))

plt.axis("off")

plt.imshow(Wc , interpolation = 'bilinear')
#creating word cloud using skull image for fake news which depict that 

#fake news are dangerous 



skull="../input/images-coud/skull-icon.png"

icon=Image.open(skull)

mask=Image.new(mode="RGB",size=icon.size, color=(255,255,255))

mask.paste(icon, box=icon)



rgb_array=np.array(mask)
#Fake News

plt.figure(figsize = (15,15))

Wc = WordCloud(mask=rgb_array,max_words = 2000 , width = 1600 ,

               height = 800)



Wc.generate(" ".join(df[df.category == 1].text))

plt.axis("off")

plt.imshow(Wc , interpolation = 'bilinear')
#splitting the 

from sklearn.model_selection import train_test_split





X=df["text"] #feature 

y=df["category"] # traget



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#importing libraries to build a pipline

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

#this pipe line will take the text and vectorise it , and then TF-IDF, then fitting the model



text_clf=Pipeline([("tfidf",TfidfVectorizer()),("clf",LinearSVC())])

text_clf.fit(X_train,y_train)
#making prediction using the model

predictions=text_clf.predict(X_test)
from sklearn import metrics

print(metrics.classification_report(y_test,predictions))
#overall acuracy

print(metrics.accuracy_score(y_test,predictions))
#confusion matrix

print(metrics.confusion_matrix(y_test,predictions))