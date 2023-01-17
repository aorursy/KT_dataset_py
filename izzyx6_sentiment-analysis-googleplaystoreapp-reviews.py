#IMPORTING ALL NECESSARY LIBRARY TO BE USED IN THIS PROJECT

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

import pickle

import re

import time

from PIL import Image

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

%matplotlib inline
import os

print(os.listdir("../input"))
data = pd.read_csv('../input/googleplaystore_user_reviews.csv')
#IMPORTING THE GOOGLE PLAYSTORE DATASET, YOU CAN DOWNLOAD THIS DATASET DIRECTLY FROM KAGGLE

data.head()
#CHECKING FOR MISSING VALUES

data.isnull().sum()
data.shape
data['App'].nunique()
data.dropna(inplace=True)
data.isnull().sum()
data.shape
data.describe()
data.info()
#CONCATINATING ONLY THE NEEDED COLUMNS TO BE USED BY THE MODEL LATER

df = pd.concat([data['Translated_Review'],data['Sentiment']],axis=1)

df
data.shape
df['Sentiment'].value_counts()
#CHANGING THE TARGET COLUMNS TO NUMERICAL VALUES BETWEEN 0-3

df['Sentiment'] = [0 if i == 'Positive' else 1 if i == 'Negative' else 2 for i in df['Sentiment']]
df.head(10)
#VISUALIZING OUR TARGET VALUES

sns.countplot(df['Sentiment'])

plt.title("Plot of Sentiments")

plt.show()
df['Translated_Review'][1]
class BadPatterns:

    

    def Patterns(self):

        '''

        This class contains popluarly used Regular Expressions 

        '''

        '''

        emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 

          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',

          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 

          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',

          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',

          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 

          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

            

        '''

        

        alpha_pattern = r'[^a-zA-Z]'

        

        patterns = {alpha_pattern:' '}

        

        return patterns

    

class Preprocessor:

    '''

    Class to a generic way of cleaning text for

    different NLP task. The process employed by

    this class are - remover of unwanted expresions

    like url, usernames (following the @ symbol),

    repetive words and numbers.

    

    This class also perfrom tokenization, and lemmatization. 

    '''

    def __init__(self):

        defualt_patterns = BadPatterns()

        

        self.patterns = defualt_patterns.Patterns()

        self.stop_words = set(stopwords.words('english'))

        

    def test(self):

        

        print(f'Patterns: {self.patterns} Stop words: {self.stop_words}')





    def text_cleaner(self,texts):

        start_time = time.time()

        sentence_list = []

        texts = texts

        

        from nltk.stem import WordNetLemmatizer 



  



        wordLemm = WordNetLemmatizer() 



        

        

        #looping through the regular expressions to remove bad expressions from text

        for pattern, replacement in zip(self.patterns.keys(),self.patterns.values()):

            #Removing bad expressions

            for text in texts:



                cleaned_text = re.sub(pattern,replacement,text)

                sentence_list.append(cleaned_text)

                

            texts = sentence_list #perfroming a swap

            sentence_list = [] 

        #loops for perfroming lemmatization, and remover of stop words

        for text in texts:

            sentence = text.lower()

            sentence = sentence.split()

    

            sentence = [wordLemm.lemmatize(word) for word in sentence if word not in self.stop_words]

            sentence = ' '.join(sentence)

            sentence_list.append(sentence)

            sentence_list

            

        

        stop_time = time.time()

        execution_time = stop_time - start_time

        print(f'Cleaning Complete')

        print(f'Time Taken: {round(execution_time,3)} seconds')

        return sentence_list

             
#USING THE CLASS CREATED FOR CLEANING VALUES

p = Preprocessor()

cleaned_text = p.text_cleaner(df['Translated_Review'])

cleaned_text[0:2]
words = cleaned_text[:8000]

plt.figure(figsize = (15,15))

word_cloud  = WordCloud(max_words = 1000 , width = 1600 , height = 800,

               collocations=False).generate(" ".join(words))

plt.imshow(word_cloud,interpolation='bilinear')

plt.axis('off')

plt.show()
#DEFINING OUR DEPENDED AND INDEPENDED VARIABLES

y = df['Sentiment'].values

X = cleaned_text
#USING THE TRAIN TEST SPLIT TO SPLIT DATA INTO 70% (TRAINING) AND 30% (TESTING)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42, shuffle=True)

print(f'X_train: {len(X_train)} X_test: {len(X_test)}')

print(f'Data Spliting Completed.')                                                 
#CREATING A PIPELINE TO PROCESSING THE REVIEWS INTO O's AND 1's WITH Tf idf VECTORIZER

clf_LR = Pipeline([('tfidf',TfidfVectorizer()),

               ('clf',LogisticRegression())])
#TRAINING THE LOGISTIC MODEL

clf_LR.fit(X_train, y_train)

print(f'Fitting Model Completed.')
#USING THE TEST DATA TO EVALUATED THE MODEL CREATED

Score = clf_LR.score(X_test,y_test)

print(f'Accuracy: {Score*100}')
LR_pred = clf_LR.predict(X_test)
cm = confusion_matrix(y_test,LR_pred)

side_bar = ["Positive","Negative","Neutral"]



f, ax = plt.subplots(figsize=(10,5))

sns.heatmap(cm,annot =True, linewidth=.5, linecolor="r", fmt=".0f", ax = ax)



ax.set_xticklabels(side_bar)

ax.set_yticklabels(side_bar)

plt.show()
report = classification_report(y_test,LR_pred)

print(report)
#SAVING THE TRAINED MODEL AS A PICKLE FILE TO DISK

filename = 'model.pickle'

pickle.dump(clf_LR, open(filename, 'wb'))
#LOADING  THE MODEL THAT WAS PREVIOUSLY SAVED

with open('model.pickle', 'rb') as f:

    model = pickle.load(f)
def predict(texts,model):

    clean = Preprocessor()

    cleaned_text = clean.text_cleaner(texts)#Cleaning the text pased to the model

    

    sentiment = model.predict(texts)

        

    match = []

    for text, pred in zip(texts,sentiment):

        match.append((text,pred))#Saving the text and genetaed sentiments by the model

        

    df = pd.DataFrame(match,columns=['Reviews','Sentiments'])#Creating a dataframe to store predictions

    df = df.replace([0,1,2], ['Positive','Negative','Neutral'])

    

    return df
#USING THE SAVED MODEL FOR A REAL CASE

text = ['I should have never downloaded your app, its worthless',

            'what did you even build',

            'You killed it with this app',

       'Keep Up ALL THE GOOD WORK']



model.predict(text)



result = predict(text,model)

print(result)