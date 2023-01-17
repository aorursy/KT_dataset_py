#Importing the essential libraries

#Beautiful Soup is a Python library for pulling data out of HTML and XML files

#The Natural Language Toolkit



import requests

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer 

from bs4 import BeautifulSoup

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import random

from wordcloud import WordCloud

from html.parser import HTMLParser
text="Keras TensorFlow Pillow Numpy Pandas Data TextBlob Data_Science Requests OpenCV Matplotlib NetworkX Scikit_Learn Analytics Code Programming Bokeh BeautifulSoup Machine_Learning Luminoth PyTorch Cirq wxPython Scipy Arrow NLTK Theano "
# using split() 

# to extract words from string 

res = text.split() 
res
len(res)
a=np.random.rand(27)
# intialise data of lists. 

data = {'word':res, 'num':a} 

  

# Create DataFrame 

df = pd.DataFrame(data) 
df["num"]=df["num"]*100
df
tuples = [tuple(x) for x in df.values]
wordcloud = WordCloud(width=1400,height=1200).generate_from_frequencies(dict(tuples))
plt.subplots(figsize=(10,10))

plt.imshow(wordcloud)

text2="PYTHON ANDROID_DEVELOPMENT XML JAVA FIREBASE_AUTHENTICATION GOOGLE DATA_ANALYSIS GITHUB FRONT_END BACK_END STACK_OVERFLOW KAGGLE KERAS TENSORFLOW OPENCV PYTORCH PYGAME APACHE OPENJDK JUNIT BOKEH DEVC++ KOTLIN LEETCODE KAGGLE CODECHEF GPU INTEL ANACONDA PANDAS JAVA C++ C# DJANGO RUBY PHP HTML CSS FLASK AZURE AWS AZURE_STACK BLOCKCHAIN IOT BEAUTIFULSOUP PYTORCH "
# using split() 

# to extract words from string 

res2 = text2.split() 
len(res2)
a2=np.random.rand(46)
# intialise data of lists. 

data2 = {'word':res2, 'num':a2} 

  

# Create DataFrame 

df2 = pd.DataFrame(data2) 
df2["num"]=df2["num"]*100
tuples2 = [tuple(x) for x in df2.values]
wordcloud2 = WordCloud(width=1400,height=1200,max_font_size=200).generate_from_frequencies(dict(tuples2))
plt.subplots(figsize=(10,10))

plt.imshow(wordcloud2)

plt.axis("off")