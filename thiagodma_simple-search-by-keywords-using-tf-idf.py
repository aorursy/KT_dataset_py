import pickle, nltk, os, json, re

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from collections import Counter

nltk.download('stopwords')
#function that gets all .json paths

def getListOfFiles(dirName):

    # create a list of file and sub directories 

    # names in the given directory 

    listOfFile = os.listdir(dirName)

    allFiles = list()

    # Iterate over all the entries

    for entry in listOfFile:

        # Create full path

        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 

        if os.path.isdir(fullPath):

            allFiles = allFiles + getListOfFiles(fullPath)

        else:

            if fullPath[-5:] == '.json': allFiles.append(fullPath)

                

    return allFiles
#getting all the paths into a list

paths = getListOfFiles('/kaggle/input/CORD-19-research-challenge')
#extracts the abstract and body text of a .json file

def clean_json(path):

    #loads json file into a string

    with open(path) as f:

        dic = json.load(f)

        

    try:    

        paper_id = dic['paper_id']

    

        txt = dic['abstract'][0]['text']

    except:

        #import pdb;pdb.set_trace()

        txt= ""

    for i in range(len(dic['body_text'])):

        par = dic['body_text'][i]['text']

        txt = txt + ' ' + par

    return paper_id, txt
#this cell may take a while to run



ids = []

texts = []

for path in paths:

    paper_id, txt = clean_json(path)

    ids.append(paper_id) #keep track of the paper's id so that I can merge the dataframes eventually

    texts.append(txt)

        

ids_and_texts = pd.DataFrame(data=list(zip(ids,texts)),columns=['sha','text'])
ids_and_texts.head()
path = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

metadata = pd.read_csv(path)

metadata.head()
#this function just makes the links ready to pop in your browser

def doi_url(doi):

    if not(isinstance(doi,str)): return '-'

    return f'http://{doi}' if doi.startswith('doi.org') else f'http://doi.org/{doi}'



metadata.doi = metadata.doi.apply(doi_url)
#all articles with sha

ok = metadata.loc[~metadata.sha.isnull()]



#all articles without sha (doesn't have the full text)

not_sha = metadata.loc[metadata.sha.isnull()]
#I'll use the abstract as the text of the article.

not_sha['text'] = not_sha['abstract']



#If the article doesn't have the abstract I'll use the title

idx = not_sha.loc[not_sha.text.isnull()].index

not_sha.text.loc[idx] = not_sha.title.loc[idx]



#if it doesn't have a title I'll drop it

not_sha.drop(not_sha.text.isnull().index,inplace=True)
#now merging the metadata with the dataframe that has the full texts

full = ok.merge(ids_and_texts,on=['sha'],how='inner')

full = pd.concat((full,not_sha))

full = full.reset_index()

full.head()
#I'll use this latter

all_data = full.copy()
#making all characters lower case

full.text = full.text.apply(str.lower)
full.text.head(3)
#taking off numbers and punctuation

full.text = full.text.apply(lambda x: re.sub(r'[^a-z ]','',x))
full.text.head(3)
#this cell may also take a while to run



#vectorizing and removing stop words

tfidf = TfidfVectorizer(max_features=5000, stop_words=nltk.corpus.stopwords.words('english'))

X = tfidf.fit_transform(full.text)
#making shure all features have 0 mean and unit standard deviation

scaler = StandardScaler()

X = scaler.fit_transform(X.todense())
#getting the keywords for each text

def get_keywords(X,tfidf,k=100):

    '''

    X: is the features matrix

    tfidf: is the tfidf object used to vectorize the texts

    k: maximum number of keywords for each text

    '''



    feature_names = tfidf.get_feature_names()

    keywords = []

    ponts_tfidf = []

    for i in range(X.shape[0]):

        text_vector = X[i]

        idxs = np.array(text_vector.argsort()[-k:][::-1]).T #getting the index of the most important words (with more tfidf ponctuation)

        s=''

        for j in range(k):            

            # sometimes 100 keywords are too much, so I make sure I don't get useless words

            if text_vector[idxs[j]] != 0:

                s = s + feature_names[idxs[j]] + ','

        keywords.append(s)

    return keywords
keywords = get_keywords(X,tfidf)



#adding the new column

all_data['keywords'] = keywords



all_data.head()
def search_keyword(df, user_keywords:list):



    idxs = []

    for user_keyword in user_keywords:

        bools = df.keywords.apply(lambda x: user_keyword in x) #does this text have this keyword?

        idxs.extend(list(df.keywords.loc[bools].index)) #keeps track of the texts that have the keywords



    #counts how many of the keywords each text has

    counter = {k: v for k, v in sorted(dict(Counter(idxs)).items(), key=lambda item: item[1],reverse=True)}



    #initialize the output of the function. I'll append to this empty dataframe

    df_out = pd.DataFrame(columns=['title','abstract','doi'])

    user_keywords_text = []

    for idx,count in zip(counter.keys(),counter.values()):

        aux = df.loc[idx][['title','abstract','doi']]

        s = [user_keyword for user_keyword in user_keywords if user_keyword in df.loc[idx]['keywords']]

        s = ','.join(s)

        user_keywords_text.append(s)

        df_out = df_out.append(aux)



    df_out.reset_index(inplace=True)

    del df_out['index']

    df_out.columns = ['Title','Abstract','Link']

    df_out['Keywords'] = user_keywords_text

    return df_out
#list with all the keywords you are interested in. Only single lower case words

user_keywords = ['range','incubation','period','human','contagious','recovery']

df_out = search_keyword(all_data, user_keywords)
df_out.loc[0]
df_out.loc[1]
df_out.loc[2]
#list with all the keywords you are interested in. Only single lower case words

user_keywords = ['immune','response','immunity']

df_out = search_keyword(all_data, user_keywords)
df_out.loc[0]
df_out.loc[1]