# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import nltk
raw_data=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
raw_data.fillna('no data found',inplace=True)
articles=raw_data.abstract
title=raw_data.title
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_process(Data_frame):  
    nopunc = [char for char in Data_frame if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    tokens = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    lemma=WordNetLemmatizer()
    return [lemma.lemmatize(each,pos='v')for each in tokens]
# To Visualize most common words article tiltes are talking about 
from wordcloud import WordCloud
title_text=' '.join(title)
tokens=text_process(title)
new_text=' '.join(tokens)
word_cloud=WordCloud(width=800,height=400,background_color='white').generate(new_text)
word_cloud.to_image()
# classifying the articles in the metadata based on virus origin genetics evolution
covid=raw_data[raw_data.title.str.contains('covid',case=False)]# to check the string containing covid in metadata
origin=covid[covid.abstract.str.contains('origin',case=False)]# to check the string containing origin in covid data
genetics=covid[covid.abstract.str.contains('genetics',case=False)] # to check the string containing genetics covid data
evolution=covid[covid.abstract.str.contains('evolution',case=False)] # to check the string conatining evolution in covid data
# deatiled study on origin of covid
origin.info()
#Text preprocessing
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import re 

def text_summary_preprocess(text):
    '''it returns summary of the text on extract based summarization technique '''
    stop_words=stopwords.words('english')
    sent_token=sent_tokenize(text)# Sentence tokenization
    new_text=re.sub('[^a-zA-Z]',' ',text) #removing punctions numbers and special charecters
    word_tokens=word_tokenize(new_text) # word tokenization
    word_tokens_without_stopwords=[word for word in word_tokens if word.lower() not in stop_words]# removing stop words
    lemma=WordNetLemmatizer()
    word_with_lemma=[lemma.lemmatize(word,pos='v') for word in word_tokens_without_stopwords]# lemmatization 
    fdist=FreqDist(word_with_lemma)# to check the most repetative words 
    #tokenization of words in sentences
    sent_with_tokens=[]
    for line in sent_token:
        sent_with_tokens.append(word_tokenize(line)) 
    #Asssigning scores for tokens in sentences 
    sent_with_scores=[]
    for line in sent_with_tokens:
        score=[fdist[token] for token in line]
        sent_with_scores.append(score)
    # summation of the total score in each sentences and normalizing it
    sent_total_score=[sum(line)/len(line) for line in sent_with_scores]
    #calculating the average of the whole text
    average=sum(sent_total_score)/len(sent_total_score)
    print('The avreage of the given text is :' ,average)
    #summarization step-1
    input_from_user=float(input('Enter the summarization factor(>average):'))
    summary=[]
    for index,score in enumerate(sent_total_score):
        if score > input_from_user:
            line=' '.join(sent_with_tokens[index])
            summary.append(line)     
    summary_in_text=' '.join(summary)
    return summary_in_text
origin_text=' '.join(origin.abstract)
origin_summary=text_summary_preprocess(origin_text)
origin_word_cloud=WordCloud(width=1000,height=500,background_color='black').generate(origin_summary)
origin_word_cloud.to_image() 
# To classify the articles belongs to virus genetics  eveloution
# vectorization of thearticles
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec=TfidfVectorizer(analyzer=text_process,stop_words='english')
x_train_df=tf_vec.fit_transform(articles)
# To find the nearest articles in the metadata related to genetics of the virus
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=10,p=2)# finidng 10 nearest articles 
nn_fitted=nn.fit(x_train_df)

# Checking realated articles 
genetics_match_vec=tf_vec.transform([genetics.abstract])
score,neigbours=nn_fitted.kneighbors(genetics_match_vec)
genetics_from_raw=raw_data.iloc[neigbours[0]]
genetics_from_raw=genetics_from_raw.abstract
# Summarization of the all the 10 articles found using nearest neigbours
genetics_raw_text=' '.join(genetics_from_raw)
genetics_summary=text_summary_preprocess(genetics_raw_text)
#To check the most common words used in the summary
genitcs_word_cloud=WordCloud(width=1000,height=500,background_color='black').generate(genetics_summary)
genitcs_word_cloud.to_image()
# Summarization of the all the 10 articles found using nearest neigbours
evolution_raw_text=' '.join(evolution.abstract)
evolution_summary=text_summary_preprocess(evolution_raw_text)
# chekcing the realated articles for the evolution of the virus
evolution_match_vec=tf_vec.transform([evolution_summary])
score,neigbours=nn_fitted.kneighbors(evolution_match_vec)
evolution_from_raw=raw_data.iloc[neigbours[0]]
evolution_from_raw=evolution_from_raw.abstract

evolution_summary=' '.join(evolution_from_raw)

#To check the most common words used in the summary
evolution_word_cloud=WordCloud(width=1000,height=500,background_color='black').generate(evolution_summary)
evolution_word_cloud.to_image()
# Finalisation of the study

# Lowest common subsequence

def lcs(a,b):
    """
    returns longest common subsequence
    """
    
    lengths = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i][j+1], 
                                        lengths[i+1][j])
    dist=lengths[i+1][j+1]
    
    i = len(a) # rows
    j = len(b) # cols
    
    # Create a character array to store the lcs string
    index = lengths[-1][-1]
    lcs_string = [""] * (index+1) 
    lcs_string[index] = "" 
  
    # Start from the right-most-bottom-most corner and 
    # one by one store characters in lcs[] 
    while i > 0 and j > 0: 
  
        # If current character in X[] and Y are same, then 
        # current character is part of LCS 
        if a[i-1] == b[j-1]: 
            lcs_string[index-1] = a[i-1] 
            i-=1
            j-=1
            index-=1
  
        # If not same, then find the larger of two and 
        # go in the direction of larger value 
        elif lengths[i-1][j] > lengths[i][j-1]: 
            i-=1
        else: 
            j-=1
        
    string=''.join(lcs_string)
    return string
from nltk.tokenize import sent_tokenize
origin_token=sent_tokenize(origin_summary)
genetics_token=sent_tokenize(genetics_summary)
evolution_token=sent_tokenize(evolution_summary)
def auto_suggest_query():
    Input=input('Would like to know about ("origin","genetics","evolution" type any of the option) : ').lower()
    query=input('shoot your query : ')
    if Input=='origin':
        distances = []
        for each in origin_token:
            distances.append(len(lcs(query, each)))
        if len(set(distances)) == 1: 
            return "Provide Further Information"
        else:
            return origin_token[distances.index(max(distances))]
        
    elif Input=='genetics':
        distances = []
        for each in genetics_token:
            distances.append(len(lcs(query, each)))
        if len(set(distances)) == 1: 
            return "Provide Further Information"
        else:
            return genetics_token[distances.index(max(distances))]
        
    elif Input=='evolution':
        distances = []
        for each in evolution_token:
            distances.append(len(lcs(query, each)))
        if len(set(distances)) == 1: 
            return "Provide Further Information"
        else:
            return evolution_token[distances.index(max(distances))]
auto_suggest_query()
