##Libraries

import numpy as np 

import pandas as pd 

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt
#Geting the Table of Studies

## a table of 18 columns and 63571 entries

articles=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') 

articles.shape
####Cleaned the TABLE for readability

Art1= articles[['title','publish_time','journal','url','abstract','doi','cord_uid']]

#Made a copy to work with

Articles1=Art1.copy()
#separate each word in the column: abstract, for browsing

Articles1['words'] = Articles1.abstract.str.strip().str.split('[\W_]+')



#separate words in the abstract column and create a new column

Articles1 = Articles1[Articles1.words.str.len() > 0]

Articles1.head(3)
# saving the Table (dataframe) above

Articles1.to_csv('Articles.csv') 
##1 Human immune response to COVID-19

#looking through the abstracts  

## 

##TABLE OF abstracts related to COVID 

COVID=Articles1[Articles1['abstract'].str.contains('COVID')]

COVID.shape

#5443 entries with ABSTRACTS contain the term COVID.
# saving the dataframe above

COVID.to_csv('COVID_ArticleAbstracts.csv') 
##Looking among COVID articles for immune response



ImmResp=COVID[COVID['abstract'].str.contains('immune response')]

ImmResp.shape

##There are 124 COVID articles with abstracts 

##that include the phrase 'immune response'

# 'human immune response' search showed NO articles.
Human=ImmResp[ImmResp['abstract'].str.contains('human')]

Human.shape

#36 article abstracts found
Human
# saving the TABLE above as a 

#table-answer to task 1

Human.to_csv('COVID_ArticleAbstracts_Human_Immune_Response.csv') 
#looking through the abstracts  

## 

##TABLE OF abstracts related to mutation 

Mutation=Articles1[Articles1['abstract'].str.contains('mutation')]

Mutation.shape

#2105 article abstracts found
coronaMut=Mutation[Mutation['abstract'].str.contains('corona')]

coronaMut.shape

##among the 2105, 580 include the term corona
COVIDMut=Mutation[Mutation['abstract'].str.contains('COVID')]

COVIDMut.shape

##among the 2105 mutation abstracts, 

#there are 95 article abstracts that include the term COVID
# saving the TABLE above as a 

#table-answer to task 2

COVIDMut.to_csv('COVID_ArticleAbstracts_COVID_mutation.csv') 
#to omit:

symbols1='!@#$%&*.,?"-'

ignoreThese=['background', 'abstract',

             'our','this','the',

             'objective','since', 'name',

            'word', 'words', 'and',

            'summary', 'study', 'dtype',

            'goal']



for char in symbols1:

        words1=COVIDMut['words'].replace(char,' ')

#lower case all words

words1=str(words1)

words1=words1.lower()



#ignore words

for item in ignoreThese:

        words1=words1.replace(item, ' ') 

        

wordcloud = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words1)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
##Using the COVID table of article abstracts

#to find studies about potential adaptation

Adapt=COVID[COVID['abstract'].str.contains('adaptation')]

Adapt.shape
Adapt
# saving the TABLE above as a 

#table-answer to task 3

Adapt.to_csv('COVID_ArticleAbstracts_COVID_adaptation.csv')
##4 Are there studies about phenotypic change?



#looking through the abstracts  

## 

##TABLE OF abstracts related to phenotypic change 

Pheno=Articles1[Articles1['abstract'].str.contains('phenotypic change')]

Pheno.shape

#There are 18 among all the article abstracts
#what about COVID-specific abstracts?

Pheno2=COVID[COVID['abstract'].str.contains('phenotypic change')]

Pheno2.shape

# None
# saving the TABLE Pheno 

#table-answer to task 4

Pheno.to_csv('ArticleAbstracts_studies_on_phenotypic_change.csv')
#Looking through COVID-specific abstracts

Evo=COVID[COVID['abstract'].str.contains('evolve')]

Evo.shape
#searching through those article abstracts with COVID term

VirEvo=COVID[COVID['abstract'].str.contains('virus evolve')]

VirEvo.shape

#2 results
VirEvo
# saving the TABLE above 

#table-answer to task 5

VirEvo.to_csv('ArticleAbstracts_COVID_virus_evolve.csv')
##In those COVID-related article abstracts..

Genetic=COVID[COVID['abstract'].str.contains('genetic')]

Genetic.shape

#..there are 146 abstracts with the term genetic
GeneticV=COVID[COVID['abstract'].str.contains('genetic variation')]

GeneticV.shape
# saving the TABLE above 

#table-answer to task 6

GeneticV.to_csv('ArticleAbstracts_COVID_genetic_variation.csv')
trans=COVID[COVID['abstract'].str.contains('transmission')]

trans.shape

#957 article abstracts with the term transmission
##searching through the above dataframe

##for specifics

models=trans[trans['abstract'].str.contains('model')]

models.shape

##345 abstract articles about COVID transmission with the term model
##searching through the above 

##for PREDICTIONS

pred=models[models['abstract'].str.contains('predict')]

pred.shape
# saving the TABLE above 

#table-answer to task 7

pred.to_csv('ArticleAbstracts_COVID_transmission_model_predict.csv')
#to omit:

#symbols1='!@#$%&*.,?"-'

#ignoreThese=['background', 'abstract',

            # 'our','this','the','objective','since', 'name',

            #'word', 'words', 'and','summary', 'study']



for char in symbols1:

        words2=pred['words'].replace(char,' ')

#lower case all words

words2=str(words2)

words2=words2.lower()



#ignore words

for item in ignoreThese:

        words2=words2.replace(item, ' ') 

        

wordcloud = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words2)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#searching through those article abstracts with COVID term

SerialInt=COVID[COVID['abstract'].str.contains('virus evolve')]

SerialInt.shape

#found 2
SerialInt
# saving the TABLE above 

#table-answer to task 8

SerialInt.to_csv('ArticleAbstracts_COVID_Serial_Interval.csv')
#searching through ALL article abstracts for qualitative frameworks

Qual=Articles1[Articles1['abstract'].str.contains('qualitative')]

Qual.shape

#391 found
#searching through the above for: Framework

QFrame=Qual[Qual['abstract'].str.contains('framework')]

QFrame.shape

#23 article abstracts mention Qualitative Framework
# saving the TABLE above 

#table-answer to task 9

QFrame.to_csv('ArticleAbstracts_Qualitative_Framework.csv')