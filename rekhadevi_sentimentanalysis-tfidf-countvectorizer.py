import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

%matplotlib inline
#Reading data files first

train=pd.read_csv('../input/movie-reviews-data/train.csv')

#train

test=pd.read_csv('../input/movie-reviews-data/test.csv')

#test
#Checking datatypes,no. of elements and empty data columns

train.dtypes

train.shape
train.info() #No null entry in training dataset
test.shape
test.info() #No null entry in test dataset
train.columns
# Creating a new column having sentiment names against their Sentiment Label

train['Sentiment_Name']= train['Sentiment'].replace({0:"Negative",1:"Somewhat Negative",2:"Neutral",3:'Somewhat Positive',4:'Positive'})

train.head()
# There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks

#sns.axes_style()

#The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.By default noetbook is used.
sns.set_style("ticks")

sns.set_context("talk")

#style.use('seaborn-bright')

plt.figure(figsize=(10,7))

sns.countplot("Sentiment_Name",data= train)

plt.title('Seniment Types')
# Sentiments vs Pharses Count

senti_phrase=train.groupby(['Sentiment'])['PhraseId'].count().sort_values(ascending=False)

print("Phrases count wr.t various Sentiments:\n ",senti_phrase)
# Plotting Sentiment labels vs  Pharses Count

sns.set_style("ticks")

sns.set_context("talk")

x=senti_phrase.index

y=senti_phrase.values

plt.figure(figsize=(10,7))

sns.barplot(x,y,color='r')

plt.xlabel('Sentiment Labels')

plt.ylabel('Phrases Count')

plt.title('Sentiments vs Phrases Used')
# Sentiment label-vs-Sentences

senti_sentence=train.groupby(['Sentiment'])['SentenceId'].nunique().sort_values(ascending=False)

senti_sentence
# Plotting Sentiment-wise Sentences Count

sns.set_style("ticks")

sns.set_context("talk")

x1=senti_sentence.index

y1=senti_sentence.values

plt.figure(figsize=(10,7))

sns.barplot(x1,y1,color='c')

plt.xlabel('Sentiment Labels')

plt.ylabel('Sentences Count')

plt.title('Sentiments vs Sentences Count')
# Plotting relationship between Sentiment labels vs Sentences and Pharses used

sns.set_style("darkgrid")

sns.set_context("talk")

plt.figure(figsize=(15,10))



x=senti_phrase.index

y=senti_phrase.values



plt.subplot(2,1,1)

plt.bar(x,y,color='c')

plt.title('Sentiment vs Phrases Used')



x1=senti_sentence.index

y1=senti_sentence.values



plt.subplot(2,1,2)

plt.bar(x1,y1,color='y')

plt.title('Sentiment vs Sentences')
# Sentences vs Phrases Count

sentence_phrase=train.groupby(['SentenceId'])['Phrase'].count().sort_values(ascending=False).head(20)

print("No. of phrases used per sentence:\n ",sentence_phrase)
# Most frequently used Words in Phrases
from wordcloud import WordCloud, STOPWORDS
corpus = ' '.join(train['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='white', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
# To create a image of WORDCLOUD

wordcloud.to_file("Frequent Phrases Words.png")
# Most frequently used Words in Phrases with Negative Sentiment

neg_sent_record= train[train['Sentiment']==0]

neg_sent_record.head() # Record of negative sentiment phrases



#Plotting Wordcloud for Negative Sentiment

corpus = ' '.join(neg_sent_record['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='white', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#### Top phrases words used having Somewhat Negative Sentiments(Sentiment label-1)

smwtneg_sent_record= train[train['Sentiment']==1]

smwtneg_sent_record.head() # Record of negative sentiment phrases



#Plotting Wordcloud for Somewhat positive Sentiment

corpus = ' '.join(smwtneg_sent_record['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='white', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#### Top phrases words used having Neutral Sentiments(Sentiment label-2)

neutral_sent_record= train[train['Sentiment']==2]

neutral_sent_record.head() # Record of neutral sentiment phrases



#Plotting Wordcloud for Somewhat positive Sentiment

corpus = ' '.join(neutral_sent_record['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='white', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#### Top phrases words used having Somewhat positive Sentiments(Sentiment label-3)

smwtpos_sent_record= train[train['Sentiment']==3]

smwtpos_sent_record.head()



#Plotting Wordcloud for Somewhat positive Sentiment

corpus = ' '.join(smwtpos_sent_record['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='white', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#### Top phrases words used having Positive Sentiments(Sentiment label-4)

pos_sent_record= train[train['Sentiment']==4]

pos_sent_record.head()



#Plotting Wordcloud for Somewhat positive Sentiment

corpus = ' '.join(pos_sent_record['Phrase'])

corpus = corpus.replace('.', '. ')

wordcloud= WordCloud(stopwords=STOPWORDS,background_color='black', width=2400,height=2000,).generate(corpus)

plt.figure(figsize=(15,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#Input dataset

#for Training

x_trn=train.loc[:,'Phrase'] # Independent Variable

#Output Data

y_trn= train.loc[:,'Sentiment'] # Dependent Variable



#for Test

x_tst=test.loc[:,'Phrase']# Independent Variable
# Removing Punctuations and Stop words from input data

import nltk

from nltk import word_tokenize  #NLTK is a leading platform for building Python programs to work with human language data.

from nltk.corpus import stopwords

stopwrds=stopwords.words('english')

#print("Stopwords:\n ",stopwrds)

import string



#first we define a function to remove punctuation

def text_cleaning(x_trn):

    textseparate = [char for char in trn if char not in string.punctuation]

    print(textseparate)

    textseparate=''.join(textseparate)

    print(textseparate)

    print(textseparate.split())

  # now we need to remove stopwords from our column  

    return [word for word in textseparate.split() if word.lower() not in stopwords.words('english')]

 # now we have cleaned title - no punctuation and stopword in it now

    

# Cleaning the test data in the same way

#Removing punctuation

def text_cleaning(x_tst):

    textseparate = [char for char in trn if char not in string.punctuation]

    print(textseparate)

    textseparate=''.join(textseparate)

    textseparate

    textseparate.split()

  # now we need to remove stopwords from our column  

    return [word for word in textseparate.split() if word.lower() not in stopwords.words('english')]

 # now we have cleaned title - no punctuation and stopword in it now
#Using TfIfd to give weightage to each word in input dataset

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf= TfidfVectorizer()

trn_data1 = tfidf.fit_transform(x_trn)



#tranforming x_test

tst_data1=tfidf.transform(x_tst)

tst_data1



# Converting trn_data and tst_data to array

np.array(trn_data1)

print(trn_data1) #Final input training data



np.array(tst_data1)#Final input training data

print(tst_data1)
from sklearn.naive_bayes import MultinomialNB

multiNB= MultinomialNB()

learning= multiNB.fit(trn_data1,y_trn)



# Predicting Output

y_pred_mult=learning.predict(tst_data1)

print ("Prediction of Sentiment Labels through Multinomial Algorithm:",  y_pred_mult)
from sklearn.naive_bayes import BernoulliNB

bern= BernoulliNB()



learning=bern.fit(trn_data1,y_trn)



y_pred_bern = bern.predict(tst_data1)

print ("Prediction of Sentiment Labels through Bernoullis Algorithm:",  y_pred_bern)
#Using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

count_vector=CountVectorizer()

trn_data2 = count_vector.fit_transform(x_trn)



#tranforming x_test

tst_data2=count_vector.transform(x_tst)

tst_data2



# Converting trn_data and tst_data to array

np.array(trn_data2)

#print(trn_data2) #Final input training data



np.array(tst_data2)#Final input training data

#print(tst_data2)
from sklearn.naive_bayes import MultinomialNB

multiNB1= MultinomialNB()

learning_cv_m= multiNB1.fit(trn_data2,y_trn)



# Predicting Output

y_pred_mult2=learning_cv_m.predict(tst_data2)

print ("Prediction of Sentiment Labels through Multinomial Algorithm :",  y_pred_mult2)
from sklearn.naive_bayes import BernoulliNB

bern1= BernoulliNB()



learning_cv_b=bern1.fit(trn_data2,y_trn)



y_pred_bern2 = learning_cv_b.predict(tst_data2)

print ("Prediction of Sentiment Labels through Bernoullis Algorithm:",  y_pred_bern2)
# Reading Sample Submission File

output_file=pd.read_csv('../input/movie-reviews-data/SampleSubmission.csv')

output_file.head(10)
#Tfidf Algorithm Outputs

#Adding our Algorithm outputs using Output File in new column

output_file['Sentiment']= y_pred_mult # Extracting output from Mutinomial algorithm

output_file



# Converting this output sheet in a csv file

output_file.to_csv('Submission_Mutlinomial through_TfIdf.csv')
output_file['Sentiment']= y_pred_bern # Extracting output from Bernoullis algorithm



# Converting this output sheet in a csv file

output_file.to_csv('Submission_Bernoulli through_TfIdf.csv')
# Count Vectorizer Algorithm Outputs

#Adding our Algorithm outputs using Output File in new column

output_file['Sentiment']= y_pred_mult2 # Extracting output from Mutinomial algorithm

output_file

# Converting this output sheet in a csv file

output_file.to_csv('Submission_Mutlinomial through_CountVectorizer.csv')
output_file['Sentiment']= y_pred_bern2 # Extracting output from Bernaullis algorithm

output_file.head()



# Converting this output sheet in a csv file

output_file.to_csv('Submission_Bernoulli through_CountVectorizer.csv')