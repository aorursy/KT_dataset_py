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
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy

#read data from xlsx
excel_dataframe = pd.read_excel('../input/unstructured-data/Unstructured Data Japanese.xls')

#read only required columns
excel_dataframe = excel_dataframe[['Sno', 'Kraft Super bowl data']]

#extract Katakana text Translated column and convert it to string format
tweets = excel_dataframe['Kraft Super bowl data'].to_string(index=False)

#manual data cleaning
tweets = tweets.replace('\\n', '')
tweets = tweets.replace('\\t', '')
tweets = tweets.replace(' .', '.')
tweets = tweets.replace('  .', '.')
tweets = tweets.replace(' . ', '. ')
tweets = tweets.replace('..', '')
tweets = tweets.replace('...', '. ')
tweets = tweets.replace(';-)', '')
tweets = tweets.replace('!', '')
tweets = tweets.replace('!!', '')
tweets = tweets.replace('-', ' ')
tweets = tweets.replace(': ', ' ')
tweets = tweets.replace('#', '')
tweets = tweets.replace('  ', ' ')

#tokenize the tweets into sentences
sentence_tokens = sent_tokenize(tweets)

#convert all the lettrs to lowercase
lower_tokens = [token.lower() for token in sentence_tokens]

#apply Counter on lowercase tokens
counter = Counter(lower_tokens)

print('Most common phrases among all tweets are :')
#number to print serial number
number=1

#extract to 10 phrases
top_10_phrases = counter.most_common(20)
for phrase, count in top_10_phrases :
    print(str(number),': ', phrase, str(count))
    number+=1

#create list of tweet data
tweets = list(excel_dataframe['Kraft Super bowl data'])

#initialize lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
data=[]

for i in range(len(tweets)):
    #convert tweets to lowercase
    sentence = str(tweets[i]).lower()
    
    #tokenize sentences to words
    wordslist = word_tokenize(sentence)
    
    #keep only aplhabet tokens
    alpha = [word for word in wordslist if word.isalpha()]
    
    #remove stopwords from alpha
    alpha_nostop = [word for word in alpha if word not in stopwords.words('english')]
    
    #Lemmatize the words to its root form
    alpha_nostop_lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in alpha_nostop]
    
    #append the lemmatized words list to data
    data.append(alpha_nostop_lemmatized)

#cleaned tweets list set to empty
cleaned_tweets=[]

#converts the list(list()) to list(strings)
for tweets in data :
    string = str(tweets[:1]).replace(',','').replace("'",'').replace('[[','"').replace(']]','"')    
    cleaned_tweets.append(string)

#extract named entity list from spacy for theme selection
nlp = spacy.load('en')
named_entities = nlp(str(cleaned_tweets))

#print 10 most common named entities
named_entity_list = []
for entity in named_entities.ents :
    named_entity_list.append(entity.text)

print('\nNamed entity list for theme selection')
counter = Counter(named_entity_list)
print(counter.most_common(20))

#instantiate count vectorizer with english stopwords
count_vectorizer = CountVectorizer(stop_words='english')

#perform fit_transform on cleaned_tweets to get the Term Document Metrix
csr_matrix = count_vectorizer.fit_transform(cleaned_tweets)
tdm = csr_matrix.toarray()

#extract tokens
tokens = count_vectorizer.get_feature_names()

#export Term Document Metrix to dataframe
df = pd.DataFrame(tdm.transpose(), index=tokens)
print("\n",df)

#write the Term Document Metrix to the csv file
df.to_csv("DTMJapanese.csv")
print('Document Term Metrix written to .csv file successfully.')