import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from nltk.sentiment.vader import SentimentIntensityAnalyzer
df = pd.read_csv('../input/amazon/Amazon_Instant_Video_5.csv')
print('Dataset 1 shape {}'.format(df.shape))
print('-Database examples-')
print(df.iloc[:550, :])
text = df['reviewText']
text = text.head(550)
print(text)
overall = df['overall']
overall = overall.head(550)
print(overall)
dfList = text.tolist()
tricky_sentences = []
for i in range(len(dfList)):
    tricky_sentences.append(dfList[i])
print(tricky_sentences)

#sentences.extend(tricky_sentences)
allOverall = []
allScore = []
sid = SentimentIntensityAnalyzer()
counter = 0
for sentence in tricky_sentences:
    #print(sentence)
    ss = sid.polarity_scores(str(sentence))
    #for k in sorted(ss):
    if float(ss['compound']) >0:
        #round up numbers to 1 which are greater than 0
        ss['compound'] = '1'
    elif float(ss['compound']) < 0:
        #round down numbers to -1 which are less than 0
        ss['compound'] = '-1'
    elif float(ss['compound']) == 0:
        #change the data type of 0
        ss['compound'] = '0'
    #append scores to a list
    allScore.append(ss['compound'])
    #append overall stars to a list
    allOverall.append(overall[counter])
        #print('{0}: {1}, '.format(k, ss[k]), end='')
    counter = counter + 1
    
print(allScore)
print(allOverall)
#counter for positive scores
Pcounter = 0
#counter for negative scores
Ncounter = 0

for i in range(len(allScore)):
    if int(allScore[i]) > 0:
        #counting the positive scores
        Pcounter = Pcounter +1
    elif int(allScore[i]) < 0:
        #counting the negative scores
        Ncounter = Ncounter +1

print("Postive numbers :  " + str(Pcounter))
print("Negtive numbers :  " + str(Ncounter))

#td=(SPd/(SPd+SNd))*RS
td = (Pcounter/(Pcounter+Ncounter))*5

print("Mean :  " + str(td))
        
df = pd.read_csv('../input/amazon-products/Baby_5.csv')
print('Dataset 1 shape {}'.format(df.shape))
print('-Database examples-')
print(df.iloc[:550, :])
text = df['reviewText']
text = text.head(550)
print(text)
overall = df['overall']
overall = overall.head(550)
print(overall)
dfList = text.tolist()
tricky_sentences = []
for i in range(len(dfList)):
    tricky_sentences.append(dfList[i])
print(tricky_sentences)
#sentences.extend(tricky_sentences)
allOverall = []
allScore = []
sid = SentimentIntensityAnalyzer()
counter = 0
for sentence in tricky_sentences:
    #print(sentence)
    ss = sid.polarity_scores(str(sentence))
    #for k in sorted(ss):
    if float(ss['compound']) >0:
        #round up numbers to 1 which are greater than 0
        ss['compound'] = '1'
    elif float(ss['compound']) < 0:
        #round down numbers to -1 which are less than 0
        ss['compound'] = '-1'
    elif float(ss['compound']) == 0:
        #change the data type of 0
        ss['compound'] = '0'
    #append scores to a list
    allScore.append(ss['compound'])
    #append overall stars to a list
    allOverall.append(overall[counter])
        #print('{0}: {1}, '.format(k, ss[k]), end='')
    counter = counter + 1
    
print(allScore)
print(allOverall)
#counter for positive scores
Pcounter = 0
#counter for negative scores
Ncounter = 0

for i in range(len(allScore)):
    if int(allScore[i]) > 0:
        #counting the positive scores
        Pcounter = Pcounter +1
    elif int(allScore[i]) < 0:
        #counting the negative scores
        Ncounter = Ncounter +1

print(Pcounter)
print(Ncounter)

#td=(SPd/(SPd+SNd))*RS
td = (Pcounter/(Pcounter+Ncounter))*5

print(td)
def sentimentAnalysis(name, path):
    df = pd.read_csv(path)
    #print('Dataset 1 shape {}'.format(df.shape))
    #print('-Database examples-')
    #print(df.iloc[:550, :])
    
    text = df['reviewText']
    text = text.head(550)
    #print(text)
    
    overall = df['overall']
    overall = overall.head(550)
    #print(overall)
    
    dfList = text.tolist()
    tricky_sentences = []
    for i in range(len(dfList)):
        tricky_sentences.append(dfList[i])
    #print(tricky_sentences)
    
    #sentences.extend(tricky_sentences)
    allOverall = []
    allScore = []
    sid = SentimentIntensityAnalyzer()
    counter = 0
    for sentence in tricky_sentences:
        #print(sentence)
        ss = sid.polarity_scores(str(sentence))
        #for k in sorted(ss):
        if float(ss['compound']) >0:
            #round up numbers to 1 which are greater than 0
            ss['compound'] = 'positive'
        elif float(ss['compound']) < 0:
            #round down numbers to -1 which are less than 0
            ss['compound'] = 'negative'
        elif float(ss['compound']) == 0:
            #change the data type of 0
            ss['compound'] = 'neutral'
        #append scores to a list
        allScore.append(ss['compound'])
        #append overall stars to a list
        allOverall.append(overall[counter])
            #print('{0}: {1}, '.format(k, ss[k]), end='')
        counter = counter + 1
    
    print(name)
    print(allScore)
    #print(allOverall)
def printing(name, path):
    df = pd.read_csv(path)
    #print('Dataset 1 shape {}'.format(df.shape))
    #print('-Database examples-')
    #print(df.iloc[:550, :])
    
    text = df['reviewText']
    text = text.head(550)
    #print(text)
    
    overall = df['overall']
    overall = overall.head(550)
    #print(overall)
    
    dfList = text.tolist()
    tricky_sentences = []
    for i in range(len(dfList)):
        tricky_sentences.append(dfList[i])
    #print(tricky_sentences)
    
    #sentences.extend(tricky_sentences)
    allOverall = []
    allScore = []
    sid = SentimentIntensityAnalyzer()
    counter = 0
    for sentence in tricky_sentences:
        #print(sentence)
        ss = sid.polarity_scores(str(sentence))
        #for k in sorted(ss):
        if float(ss['compound']) >0:
            #round up numbers to 1 which are greater than 0
            ss['compound'] = 'positive'
        elif float(ss['compound']) < 0:
            #round down numbers to -1 which are less than 0
            ss['compound'] = 'negative'
        elif float(ss['compound']) == 0:
            #change the data type of 0
            ss['compound'] = 'neutral'
        #append scores to a list
        allScore.append(ss['compound'])
        #append overall stars to a list
        allOverall.append(overall[counter])
            #print('{0}: {1}, '.format(k, ss[k]), end='')
        counter = counter + 1
    
    #print(allScore)
    #print(allOverall)
    
    #counter for positive scores
    Pcounter = 0
    #counter for negative scores
    Ncounter = 0

    for i in range(len(allScore)):
        if allScore[i] is 'positive':
            #counting the positive scores
            Pcounter = Pcounter +1
        elif allScore[i] is 'negative':
            #counting the negative scores
            Ncounter = Ncounter +1

    #print(Pcounter)
    #print(Ncounter)

    #td=(SPd/(SPd+SNd))*RS
    td = (Pcounter/(Pcounter+Ncounter))*5
    
    #print(td)
    
    bts = df['overall'].mean()
    #print(bts)
    print("-----  brand  ------, ----  trust  ----, ----  rating  -----")
    
    print('{0}, {1}, {2}'.format(name,td,bts))


    
printing('Amazon_Instant_Video','../input/amazon/Amazon_Instant_Video_5.csv')
sentimentAnalysis('Amazon_Instant_Video','../input/amazon/Amazon_Instant_Video_5.csv')
printing('Baby','../input/amazon-products/Baby_5.csv')
printing('Digital_Music','../input/amazon-products/Digital_Music_5.csv')
printing('Musical_Instruments','../input/amazon-products/Musical_Instruments_5.csv')
printing('Patio_Lawn_and_Garden','../input/amazon-products/Patio_Lawn_and_Garden_5.csv')
printing('Automotive','../input/amazon-products-2/Automotive_5.csv')
printing('Grocery_and_Gourmet_Food','../input/amazon-products-2/Grocery_and_Gourmet_Food_5.csv')
printing('Apps_for_Android','../input/amazon-products-3/Apps_for_Android_5.csv')
printing('Beauty','../input/amazon-products-3/Beauty_5.csv')
printing('Office_Products','../input/amazon-products-3/Office_Products_5.csv')
printing('Pet_Supplies','../input/amazon-products-3/Pet_Supplies_5.csv')
printing('CDs_and_Vinyl','../input/amazon-products-4/CDs_and_Vinyl.csv')
printing('Cell_Phones_and_Accessories','../input/amazon-products-4/Cell_Phones_and_Accessories.csv')
printing('Clothing_Shoes_and_Jewelry','../input/amazon-products-4/Clothing_Shoes_and_Jewelry.csv')
printing('Electronics','../input/amazon-products-4/Electronics.csv')
printing('Health_and_Personal_Care','../input/amazon-products-4/Health_and_Personal_Care.csv')
printing('Home_and_Kitchen','../input/amazon-products-4/Home_and_Kitchen.csv')
printing('Movies_and_TV','../input/amazon-products-4/Movies_and_TV.csv')
printing('Sports_and_Outdoors','../input/amazon-products-4/Sports_and_Outdoors.csv')
printing('Tools_and_Home_Improvement','../input/amazon-products-4/Tools_and_Home_Improvement.csv')
printing('Toys_and_Games','../input/amazon-products-4/Toys_and_Games.csv')
printing('Video_Games','../input/amazon-products-4/Video_Games.csv')