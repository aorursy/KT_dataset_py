import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load the meta data from the CSV file using 3 columns (abstract, title, authoris)

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['abstract','title','authors'])

#drop NANs 

df=df.dropna()

#show 5 lines of the new dataframe

df.head()
import functools

from IPython.core.display import display, HTML



#tell the system how many sentences are needed

max_sentences=5



# list of list for topic words realting to tasks

tasks = [["incubation", "period", "days"],["contagious", "incubation"],["asymptomatic","transmission"],['children'],['season'],['adhesion'],['environmental'],["comorbidities","risk"],["blood type","type"],['smoking'],["common","symptoms"]]

# loop throuhg the list of lists

for search_words in tasks:

    str1=''

    # add COVID-19 to focus the search the papers have many unrealted documents

    search_words.append("COVID-19")

    # search the dataframe for all the keywords

    df1=df[functools.reduce(lambda a, b: a&b, (df['abstract'].str.contains(s) for s in search_words))]

    # remove the COVID-19 word for sentence level analysis

    search_words.pop()

    # a make a stirng of the search words to show search

    str1=' '.join(search_words)

    display(HTML('<h3>Search: '+str1+'</h3>'))

    # record how many sentences have been saved for display

    sentences_used=0

    # loop through the result of the datframe search

    for index, row in df1.iterrows():

        #break apart the absracrt to sentence level

        sentences = row['abstract'].split('. ')

        #loop through the sentences of the abstract

        for sentence in sentences:

            # missing lets the system know if all the owrds are in the sentence need scoring system upgrade

            missing=0

            #loop through the words of sentence

            for word in search_words:

                #if keyword missing change missing variable

                if word not in sentence:

                    missing=1

            # after all sentences processed show the sentences not missing keywords limit to max_sentences

            if missing==0 and sentences_used < max_sentences:

                sentences_used=sentences_used+1

                display(HTML('<b>'+sentence+'</b> - <i>'+row["title"]+'</i>, '+row["authors"]))

print ("done")