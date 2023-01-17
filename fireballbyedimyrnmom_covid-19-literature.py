#libraries

import numpy as np 

import pandas as pd 

import numpy as np
#directory of articles

df1=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

df1.head(2)
#selecting specific columns

##these include the title of the article, its abstract, date, link to the article, and authors

journals= df1[['title', 'abstract', 'publish_time', 'url', 'authors']]
#separate each word in the ABSTRACT column

journals['words'] = journals.abstract.str.strip().str.split('[\W_]+')

journals['words'].head()
#separate words in the abstract column and create a new column

abstracts = journals[journals.words.str.len() > 0]

abstracts.head(2)
#A column of individual words

rows = list()

for row in abstracts[['words']].iterrows():

    r = row[1]

    for word in r.words:

        rows.append((word))



words = pd.DataFrame(rows, columns=['word'])

words.head(2)
#changes strings to lowercase for ease of search

text1 = words.word.str.lower()
#remove punctuation

text1=text1.str.replace('[^\w\s]','')
counts = text1.value_counts()

common1=counts.head(24)
#removes common words like the, of, etc.



common1 = list(common1.index)

text1 = text1.apply(lambda x: " ".join(x for x in x.split() if x not in common1))

text1.head(20)
#number of unique words

text1.nunique()
text1.value_counts().head(20)
#looking for rows with specific term

##VACCINE RELATED articles

abstracts[abstracts['abstract'].str.contains('vaccine')]
##therapeutic RELATED articles

abstracts[abstracts['abstract'].str.contains('therapeutic')]
##COVID specific articles

lit_COVID=abstracts[abstracts['abstract'].str.contains('COVID')]

lit_COVID.head(2)
#looking for rows with specific term

##articles about monitor

abstracts[abstracts['abstract'].str.contains('monitor')]
##articles related to trace

abstracts[abstracts['abstract'].str.contains('trace')]
##articles about surveillance

surv=abstracts[abstracts['abstract'].str.contains('surveillance')]

surv.head(2)
##articles about ELISA

abstracts[abstracts['abstract'].str.contains('ELISA')]
##articles about antibodies

antib=abstracts[abstracts['abstract'].str.contains('antibodies')]

antib.head(2)
#looking for rows with specifid terms

##articles about outcome

abstracts[abstracts['abstract'].str.contains('outcome')]
#lit_COVID is the 1892 articles defined before.

#from here, we can narrow the search.

COVID_thera=lit_COVID[lit_COVID['abstract'].str.contains('therapeutic')]

# a new dataframe of COVID19 articles on therapetic

## it becomes easier to scroll through 117 articles

COVID_thera
COVID_diag=lit_COVID[lit_COVID['abstract'].str.contains('diagnosis')]

# a new dataframe of COVID19 articles on diagnosis

## it becomes easeier to scroll through 225 articles

COVID_diag
monitor1=antib[antib['abstract'].str.contains('diagnosis')]

# a new dataframe of antibodies and monitor articles 

## it becomes easeier to scroll through 232 articles



#then focus on ELISA

#this shows 71 articles

ELImon=monitor1[monitor1['abstract'].str.contains('ELISA')]

ELImon
COVID_surv=surv[surv['abstract'].str.contains('COVID')]

# a new dataframe of COVID19 articles on survellance

## it becomes easier to scroll through 76 articles

COVID_surv