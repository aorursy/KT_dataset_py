import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import warnings
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)

warnings.filterwarnings('ignore')

review = pd.read_csv("../input/datasheet/demo - Sheet1.csv")
review.shape
review.head()
review.rename(columns={'Unnamed: 0':'S No'},inplace=True)
review['length'] = review['Tweet_Text'].apply(len)
review.info()
review['Website'].unique()
review['count_word']=review["Tweet_Text"].apply(lambda x: len(str(x).split()))
review['count_stopwords']=review['Tweet_Text'].apply(lambda x:len([w for w in str(x).lower().split() if w in stopwords]))
review['count_punct']=review['Tweet_Text'].apply(lambda x:len([p for p in str(x) if p in string.punctuation]))
plt.figure(figsize=(15,12))

plt.subplot(221)
g = sns.boxplot(x=review['length'],y=review['count_word'],palette=sns.color_palette(palette="Set1"))
g.set_title("Distribution of words in each sentences by lenght", fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(222)
g1 = sns.boxplot(x=review['length'],y=review['count_stopwords'],palette=sns.color_palette(palette="dark"))
g1.set_title("Distribution of stopwords in each sentences by lenght", fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 1, hspace = 0.6,top = 0.9)
review.head()
vect=CountVectorizer(ngram_range=(1,1),analyzer='word',stop_words=stopwords,token_pattern=r'\w{1,}')
review_vect = vect.fit_transform(review['Tweet_Text'])
review_vect.get_shape()

tf_idf=TfidfVectorizer(ngram_range=(1,1),stop_words=stopwords,analyzer='word',token_pattern=r'\w{1,}')
review_tfidf=tf_idf.fit_transform(review['Tweet_Text'])
review_tfidf.get_shape()
x_train_vec,x_test_vec,y_train_vec,y_test_vec=train_test_split(review_vect,review['Tweet_Text'],train_size=0.8,random_state=100)
logit=LogisticRegression(class_weight='balanced',multi_class='multinomial',solver='lbfgs')
logit.fit(x_train_vec,y_train_vec)
logit.get_params()
predictions=logit.predict(x_test_vec)
print("Accuracy Score with count Vectorizer: {:0.3f}".format(accuracy_score(predictions,y_test_vec)))