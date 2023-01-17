import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



def search_focus(df):

    dfa = df[df['abstract'].str.contains('covid')]

    dfb = df[df['abstract'].str.contains('-cov-2')]

    dfc = df[df['abstract'].str.contains('cov2')]

    dfd = df[df['abstract'].str.contains('ncov')]

    frames=[dfa,dfb,dfc,dfd]

    df = pd.concat(frames)

    df=df.drop_duplicates(subset='title', keep="first")

    return df



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])

print ('ALL CORD19 articles',df.shape)

#fill na fields

df=df.fillna('no data provided')

#drop duplicate titles

df = df.drop_duplicates(subset='title', keep="first")

#keep only 2020 dated papers

df=df[df['publish_time'].str.contains('2020')]

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()

#show 5 lines of the new dataframe

df=search_focus(df)
import functools

from IPython.core.display import display, HTML

from nltk import PorterStemmer



#tell the system how many sentences are needed

max_sentences=10



# function to stem keywords into a common base word

def stem_words(words):

    stemmer = PorterStemmer()

    singles=[]

    for w in words:

        singles.append(stemmer.stem(w))

    return singles



# list of lists for topic words realting to tasks

display(HTML('<h1>What is known about transmission, incubation, and environmental stability?</h1>'))

tasks = [['transmission','humidity'],['effective','reproductive','number'],['surface','persist','days'],["incubation", "period", "days"],["contagious", "incubation"],["asymptomatic","transmission"],['children'],['season'],['prevention','control'],['adhesion'],['environmental'],["comorbidities"],['disease', 'model'],['phenotypic'],['immune','response'],['movement','control'],['protective','equipment'],["blood type","type"],['smoking'],["common","symptoms"]]

# loop through the list of lists

for search_words in tasks:

    str1=''

    # a make a string of the search words to print readable search

    str1=' '.join(search_words)

    search_words=stem_words(search_words)

    # add cov to focus the search the papers and avoid unrelated documents

    search_words.append("covid-19")

    # search the dataframe for all the keywords

    dfa=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]

    search_words.pop()

    search_words.append("cov")

    dfb=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]

    # remove the cov word for sentence level analysis

    search_words.pop()

    #combine frames with COVID and cov and drop dups

    frames = [dfa, dfb]

    df1 = pd.concat(frames)

    df=df.drop_duplicates()

    

    display(HTML('<h3>Task Topic: '+str1+'</h3>'))

    # record how many sentences have been saved for display

    sentences_used=0

    # loop through the result of the dataframe search

    for index, row in df1.iterrows():

        #break apart the absracrt to sentence level

        sentences = row['abstract'].split('. ')

        #loop through the sentences of the abstract

        for sentence in sentences:

            # missing lets the system know if all the words are in the sentence

            missing=0

            #loop through the words of sentence

            for word in search_words:

                #if keyword missing change missing variable

                if word not in sentence:

                    missing=1

            # after all sentences processed show the sentences not missing keywords limit to max_sentences

            if missing==0 and sentences_used < max_sentences:

                sentences_used=sentences_used+1

                authors=row["authors"].split(" ")

                link=row['doi']

                title=row["title"]

                display(HTML('<b>'+sentence+'</b> - <i>'+title+'</i>, '+'<a href="https://doi.org/'+link+'" target=blank>'+authors[0]+' et al.</a>'))

print ("done")