import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])

print (df.shape)

#drop duplicate abstracts

df = df.drop_duplicates(subset='abstract', keep="first")

#drop NANs 

df=df.dropna()

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()

#show 5 lines of the new dataframe

print (df.shape)

df.head()
from search_func_py import *



################ main program

# list of lists of search terms

search=[['after', 'recovery','positive','testing']]



for search_words in search:

    str1=''

    # a make a string of the search words to print readable version above table

    str1=' '.join(search_words)

    

    #search the dataframe for all words

    df1=search_dataframe(df,search_words)



    # analyze search results for relevance 

    df1=search_relevance(df1,search_words)



    # get best sentences

    df_table=get_sentences(df1,search_words)

    

    #convert df to html

    df_table=HTML(df_table.to_html(escape=False,index=False))

    # display search topic

    display(HTML('<h3>Search: '+str1+'</h3>'))

    #display table

    display(df_table)



print ('done')