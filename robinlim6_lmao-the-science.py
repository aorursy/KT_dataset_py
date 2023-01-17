import requests

import urllib.request

import time

import pandas as pd

import numpy as np



import pandas as pd

import nltk

import re



from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk import pos_tag, wordnet

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



import string

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')



from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import seaborn as sn

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.metrics import confusion_matrix

training_df=pd.read_csv("../input/training_set_df.csv")
training_df['QnA'] = training_df['Question'] + str(' ') + training_df['Answer']



training_df['QnA'] = training_df['QnA'].astype(str)



special_char= "[,.®'&$’\"\-()?]"

training_df['QnA'] = training_df['QnA'].apply(lambda x:x.lower())

training_df['QnA'] = training_df['QnA'].str.replace('[^\w\s]','')

training_df['QnA'] = training_df['QnA'].str.replace('nan','')



#create new column for Tokenized words

training_df['QnA_tokenized'] = training_df['QnA'].apply(nltk.word_tokenize)



training_df['QnA_tokenized'] = training_df['QnA_tokenized'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])



pd.set_option('display.max_columns', None)

training_df.head()



#stemming QnA_tokenized column

training_df['QnA_stemmed'] = training_df['QnA_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])



training_df['QnA_stemmed']= training_df['QnA_stemmed'].astype(str)



training_df.to_csv('training_finale.csv')









test_df=pd.read_csv('../input/test_set_df.csv')



test_df['QnA'] = test_df['Question'] + str(' ') + test_df['Answer']



test_df['QnA'] = test_df['QnA'].astype(str)



special_char= "[,.®'&$’\"\-()?]"

test_df['QnA'] = test_df['QnA'].apply(lambda x:x.lower())

test_df['QnA'] = test_df['QnA'].str.replace('[^\w\s]','')

test_df['QnA'] = test_df['QnA'].str.replace('nan','')



test_df['QnA'] = test_df['Question'] + str(' ') + test_df['Answer']



test_df['QnA'] = test_df['QnA'].astype(str)



special_char= "[,.®'&$’\"\-()?]"

test_df['QnA'] = test_df['QnA'].apply(lambda x:x.lower())

test_df['QnA'] =test_df['QnA'].str.replace('[^\w\s]','')

test_df['QnA'] = test_df['QnA'].str.replace('nan','')



test_df['QnA_tokenized'] = test_df['QnA'].apply(nltk.word_tokenize)



test_df['QnA_tokenized'] = test_df['QnA_tokenized'].apply(lambda x: [item for item in x if item not in stopwords.words('english') ])



#remove stopwords

test_df['QnA_tokenized'] = test_df['QnA_tokenized'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])



#stemming QnA_tokenized column

test_df['QnA_stemmed'] = test_df['QnA_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])



test_df['QnA_stemmed'] = test_df['QnA_stemmed'].astype(str)



test_df.to_csv('test_final.csv')



test_df.head()

training_df = pd.read_csv('training_finale.csv')

test_df = pd.read_csv('test_final.csv')
## Training and evaluation of Models with Training Set



Corpus = training_df



Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['QnA_stemmed'],Corpus['subreddit'],test_size=0.2, random_state = 0)



Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features=1000)

Tfidf_vect.fit(Corpus['QnA_stemmed'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)

Test_X_Tfidf = Tfidf_vect.transform(Test_X)

Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)

predictions_NB = Naive.predict(Test_X_Tfidf)

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
array = confusion_matrix(Test_Y, predictions_NB)

cm = pd.DataFrame(array, index = ['clean', 'dirty','mean','dad'], columns = ['clean', 'dirty','mean','dad'])

sn.set(font_scale=1.4)

sn.heatmap(cm, annot=True,annot_kws={"size": 16})



SVM = svm.SVC(C=0.85, kernel='linear', degree=3, gamma='auto')

SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
array = confusion_matrix(Test_Y, predictions_SVM)

cm = pd.DataFrame(array, index = ['clean', 'dirty','mean','dad'], columns = ['clean', 'dirty','mean','dad'])

sn.set(font_scale=1.4)

sn.heatmap(cm, annot=True,annot_kws={"size": 16})
Corpus1 = test_df



Tfidf_vect = TfidfVectorizer(max_features=1000)

Tfidf_vect.fit(Corpus1['QnA_stemmed'])

Test_X_Tfidf = Tfidf_vect.transform(Corpus1['QnA_stemmed'])



predictions = SVM.predict(Test_X_Tfidf)
prediction_df = pd.DataFrame(predictions, columns = ['prediction'])
prediction_df.loc[prediction_df.prediction == 0, 'predicted_category'] = 'clean' 

prediction_df.loc[prediction_df.prediction == 1, 'predicted_category'] = 'dirty' 

prediction_df.loc[prediction_df.prediction == 2, 'predicted_category'] = 'mean' 

prediction_df.loc[prediction_df.prediction == 3, 'predicted_category'] = 'dad'
prediction_df.to_csv('predictions_finale.csv')



prediction_df = pd.read_csv('predictions_finale.csv')



prediction_df.drop(prediction_df.columns[prediction_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)



predicted_categories_df= pd.concat([test_df,prediction_df], axis = 1)

predicted_categories_df.drop(predicted_categories_df.columns[predicted_categories_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

predicted_categories_df.to_csv('predicted categories_final.csv')
cat_likes_agg_df = predicted_categories_df.copy()

cat_likes_agg_df = cat_likes_agg_df.groupby('predicted_category').sum().reset_index()



cat_count_agg_df = predicted_categories_df.groupby('predicted_category').size().reset_index(name ='Count')
fig = plt.figure(figsize=(30,20))

ax = fig.add_subplot(121)

ax1 = fig.add_subplot(122)

# Set the title of the subplot

ax.set_title("Likes vs Type of Joke")

# Plot from the df

cat_likes_agg_df.plot(x='predicted_category', y='Likes', kind = 'bar', ax =ax)

ax.set_ylabel('No. of Likes')

ax.set_title ('Likes vs Category')

plt.xlabel("Joke type", fontsize = 14)

plt.ylabel("No. of Likes", fontsize = 14)



ax1.set_ylabel('Count of Jokes')

ax1.set_title ('Count vs Category')

cat_count_agg_df.plot(x='predicted_category', y='Count', kind = 'bar', ax =ax1)



plt.show()
agg_count_likes_df= pd.concat([cat_likes_agg_df,cat_count_agg_df], axis = 1)



agg_count_likes_df['Avg likes per joke'] = agg_count_likes_df['Likes']/agg_count_likes_df['Count']



agg_count_likes_df = agg_count_likes_df.loc[:,~agg_count_likes_df.columns.duplicated()]

fig = plt.figure(figsize=(30,20))

ax = fig.add_subplot(111)



ax.set_title("Avg Likes vs Type of Joke")



agg_count_likes_df.plot(x='predicted_category', y='Avg likes per joke', kind = 'bar', ax =ax)

ax.set_ylabel('No. of Likes', fontsize = 20)

ax.set_title ('Averaged No. of Likes vs Category', fontsize = 20)



plt.xlabel("Joke type", fontsize = 14)



plt.ylabel("No. of Likes", fontsize = 14)





plt.show()

from wordcloud import WordCloud
dirtyjokes_list =[]

dirtyjokes_series = training_df.copy()

dirtyjokes_series = training_df[training_df['subreddit'] == 'DirtyJokes']

dirtyjokes_series = dirtyjokes_series['QnA_tokenized']

dirtyjokes_list = dirtyjokes_series.tolist()





dadjokes_list =[]

dadjokes_series = training_df.copy()

dadjokes_series = training_df[training_df['subreddit'] == 'MeanJokes']

dadjokes_series = dadjokes_series['QnA_stemmed']

dadjokes_list = dadjokes_series.tolist()

plt.figure(figsize=(20,10))



# create wordcloud here

plt.figure(figsize=(15,8))

wc = WordCloud(width=400, height=150, background_color="white", max_words=20, relative_scaling=1.0)

desc_wordcloud = wc.generate(str(dirtyjokes_list))





plt.imshow(desc_wordcloud)

plt.axis("off")

plt.title("Wordcloud of dirty jokes", fontsize=20)

plt.show()

plt.figure(figsize=(20,10))



# create wordcloud here

plt.figure(figsize=(15,8))

wc = WordCloud(width=400, height=150, background_color="white", max_words=20, relative_scaling=1.0)

desc_wordcloud = wc.generate(str(dadjokes_list))





plt.imshow(desc_wordcloud)

plt.axis("off")

plt.title("Wordcloud of mean jokes", fontsize=20)

plt.show()