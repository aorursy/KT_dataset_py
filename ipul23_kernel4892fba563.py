#import Library
%matplotlib inline
import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import copy
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsClassifier
import datetime
from fastai.structured import add_datepart
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
df_ayam = pd.read_csv("../input/dataset-ayam.csv")
df_ikan = pd.read_csv("../input/dataset-ikan.csv")
df_kambing = pd.read_csv("../input/dataset-kambing.csv")
df_sapi = pd.read_csv("../input/dataset-sapi.csv")
df_tahu = pd.read_csv("../input/dataset-tahu.csv")
df_telur = pd.read_csv("../input/dataset-telur.csv")
df_tempe = pd.read_csv("../input/dataset-tempe.csv")
df_ayam.shape
df_ayam.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_ayam['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_ayam['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_ikan['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_kambing['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_sapi['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_tahu['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_telur['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_tempe['Title']  = df_ikan.Title.str.replace('[^a-zA-Z]', '')
df_ayam.head()
df_ikan['Ingredients']  = df_ikan.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_ikan.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_ikan['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_kambing['Ingredients']  = df_kambing.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_kambing.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_kambing['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_sapi['Ingredients']  = df_sapi.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_sapi.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_sapi['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_tahu['Ingredients']  = df_tahu.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_tahu.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_tahu['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_telur['Ingredients']  = df_telur.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_telur.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_telur['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
df_tempe['Ingredients']  = df_tempe.Ingredients.str.replace('[^a-zA-Z]', ' ')
df_tempe.head()
from wordcloud import WordCloud
plt.figure( figsize=(15,15) )
wordcloud = WordCloud(background_color='white', relative_scaling=0, normalize_plurals = True).generate(df_tempe['Ingredients'].to_string().lower())
plt.title("Sentiment Words")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
for i in range(0,30):
    s = str(df_tempe['Title'][i])+'\n\n'+str(df_tempe['Ingredients'][i])+'\n\n'+str(df_tempe['Steps'][i])+'\n\n'
    tempe = df_tempe['Title'][i]
    text_file = open(f"../working/{str(tempe)}.txt", "w")
    text_file.write("%s" % s)
    text_file.close()
