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
import requests

from bs4 import BeautifulSoup
r = requests.get("https://www.agriculture.com/")
soup = BeautifulSoup(r.content, 'html.parser')
# Retrieve all popular news links (Fig. 1)

link = []

for i in soup.find('body').find_all('a'):

    if (i['href'][0]) == 'h':

        link.append(i['href'])
link
# For each link, we retrieve paragraphs from it, combine each paragraph as one string, and save it to documents (Fig. 2)

documents = []

for i in link:

    # Make a request to the link

    r = requests.get(i)

    # Initialize BeautifulSoup object to parse the content 

    soup = BeautifulSoup(r.content, 'html.parser')

  

    # Retrieve all paragraphs and combine it as one

    sen = []

    for i in soup.find_all('a'):

        sen.append(i.get_text())

  

    # Add the combined paragraphs to documents

    documents.append(' '.join(sen))
documents[0]
d = {'links':link,'content':documents}
documents_df = pd.DataFrame(d)
documents_df.head()
print("Number of documents =",documents_df.shape[0])
# Abbreviations data created manually for better cleaning

abb_df = pd.read_csv("../input/abbreviation/abbreviations_data.csv")

# Similarly stop words data created manually

stop_df = pd.read_excel("../input/my-stopwords/my_stopwords.xlsx") 
# Converting Stopwords of type Dataframe into list type

S = stop_df['stopwords'].to_list()

# set is used for extracting the unqiue words (since it is created manually)

StopWords = list(set(S))
# Data created for Replacing the contractions

contractions_dict = {

    "what's":"what is",

    "what're":"what are",

    "who's":"who is",

    "who're":"who are",

    "where's":"where is",

    "where're":"where are",

    "when's":"when is",

    "when're":"when are",

    "how's":"how is",

    "how're":"how are",



    "i'm":"i am",

    "we're":"we are",

    "you're":"you are",

    "they're":"they are",

    "it's":"it is",

    "he's":"he is",

    "she's":"she is",

    "that's":"that is",

    "there's":"there is",

    "there're":"there are",



    "i've":"i have",

    "we've":"we have",

    "you've":"you have",

    "they've":"they have",

    "who've":"who have",

    "would've":"would have",

    "not've":"not have",



    "i'll":"i will",

    "we'll":"we will",

    "you'll":"you will",

    "he'll":"he will",

    "she'll":"she will",

    "it'll":"it will",

    "they'll":"they will",



    "isn't":"is not",

    "wasn't":"was not",

    "aren't":"are not",

    "weren't":"were not",

    "can't":"can not",

    "couldn't":"could not",

    "don't":"do not",

    "didn't":"did not",

    "shouldn't":"should not",

    "wouldn't":"would not",

    "doesn't":"does not",

    "haven't":"have not",

    "hasn't":"has not",

    "hadn't":"had not",

    "won't":"will not"

}
def DataCleaning(documents_df, StopWords, contractions_dict):

    # converting strings to lowercase

    documents_df.content.replace(to_replace='[^a-zA-Z]', value = " ", inplace=True, regex=True) 

    

    # removing words having len <=2

    documents_df.content.replace(to_replace=r'\b\w{1,3}\b', value = "", inplace=True, regex=True) 

    

    # Remove punctuations

    documents_df['content'] = documents_df['content'].str.replace('[^\w\s]','')

    

    # Removing Stopwords

    documents_df['content'] = documents_df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in StopWords))

    

    # Replacing contractions 

    documents_df.content.replace(contractions_dict, regex=True,inplace=True)

    

    return documents_df
documents_df = DataCleaning(documents_df, StopWords, contractions_dict)
documents_df.head()
from nltk.stem import WordNetLemmatizer 

  

lemmatizer = WordNetLemmatizer() 
# Doing lemmatization only for adjectives, verbs and adverbs, not for nouns

# it takes some time to run 

def NextLevelCleaning(df):

    for i in range(df.shape[0]):

        words = []

        # Tagging each word with their grammar meaning

        doc = df['content'][i].split()

        for token in doc:

            words.append(lemmatizer.lemmatize(token))

        df.iloc[i,1] = ' '.join(words)

    return df
documents_df = NextLevelCleaning(documents_df)
documents_df.head()
corpus = []

for i in range(documents_df.shape[0]):

    for j in documents_df['content'][i].split():

        if j.lower() not in corpus:

            corpus.append(j.lower())
len(corpus)
TD = {}



for word in corpus:

    positions = {}

    doc = []

    for i in range(documents_df.shape[0]):

        if word in documents_df['content'][i].split():

            positions[i+1] = (list(np.where(np.array(documents_df['content'][i].split()) == word)[0]))

    doc.append(len(positions))

    doc.append(positions)

    TD[word] = doc
TD['farm']
query = "agriculture crop"

query = [" ".join(query.split())]
query = pd.DataFrame(query,columns=['content'])
query = DataCleaning(query, StopWords, contractions_dict)
print("Cleaned Query:",query['content'][0])
documents_df.columns
for word in query['content'][0].split():

    if word in TD.keys():

        temp = []

        for i in (sorted(TD[word][1].items(), key=lambda x: len(x[1]), reverse=True)):

            temp.append(documents_df['links'][i[0]+1])

    print("Results for",word)

    for l in temp:

        print(l)