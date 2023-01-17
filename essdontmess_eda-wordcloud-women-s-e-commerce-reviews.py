import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ecom = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv", index_col=1)
ecom.head()
ecom.drop(columns='Unnamed: 0', inplace=True)
ecom.head()
print("There are total " + str(len(ecom.index.unique())) + " unique items in this dataset")
sns.set(style="darkgrid")
plt.figure(figsize= (14,5))
sns.distplot(ecom['Age'], hist_kws=dict(edgecolor="k")).set_title("Distribution of Age")
plt.figure(figsize= (14,5))
ax=sns.countplot(x='Rating', data=ecom)
ax.set_title("Distribution of Ratings", fontsize=14)

x=ecom['Rating'].value_counts()

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.figure(figsize= (14,5))
ax=sns.countplot(x='Department Name', data=ecom, order = ecom['Department Name'].value_counts().index)
ax.set_title("Reviews per Department", fontsize=14)
ax.set_ylabel("# of Reviews", fontsize=12)
ax.set_xlabel("Department", fontsize=12)

x=ecom['Department Name'].value_counts()

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
ecom.info()
ecom.isnull().sum()
ecom.dropna(subset=['Review Text'], inplace=True)
ecom['Length'] = ecom['Review Text'].apply(len)
ecom.head()
d = sns.FacetGrid(ecom, col='Rating')
d.map(plt.hist,'Length',bins=30)
plt.figure(figsize= (14,5))
sns.boxplot(x='Rating', y='Length', data=ecom, palette='rainbow')
import re

def clean_data(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    words = letters_only.lower().split()                            
    return( " ".join( words ))    
from wordcloud import WordCloud, STOPWORDS
stopwords= set(STOPWORDS)|{'skirt', 'blouse','dress','sweater', 'shirt','bottom', 'pant', 'pants' 'jean', 'jeans','jacket', 'top', 'dresse'}

def create_cloud(rating):
    x= [i for i in rating]
    y= ' '.join(x)
    cloud = WordCloud(background_color='white',width=1600, height=800,max_words=100,stopwords= stopwords).generate(y)
    plt.figure(figsize=(15,7.5))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()
rating5= ecom[ecom['Rating']==5]['Review Text'].apply(clean_data)
create_cloud(rating5)
rating4= ecom[ecom['Rating']==4]['Review Text'].apply(clean_data)
create_cloud(rating4)
rating3= ecom[ecom['Rating']==3]['Review Text'].apply(clean_data)
create_cloud(rating3)
rating2= ecom[ecom['Rating']==2]['Review Text'].apply(clean_data)
create_cloud(rating2)
rating1= ecom[ecom['Rating']==1]['Review Text'].apply(clean_data)
create_cloud(rating1)
