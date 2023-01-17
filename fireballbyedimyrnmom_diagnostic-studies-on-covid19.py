##Libraries

import numpy as np 

import pandas as pd 
#Geting the Table of Studies

## a table of 18 columns and 63571 entries

studies=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') 

studies.shape
####Cleaned the dataframe (TABLE) above for readability

Studies1= studies[['title','publish_time','journal','url','abstract','doi','cord_uid']]

#Make a copy to work with

StudyAbs=Studies1.copy()
#separate each word in the column: abstract for browsing

StudyAbs['words'] = StudyAbs.abstract.str.strip().str.split('[\W_]+')
#separate words in the abstract column and create a new column

Abstracts1 = StudyAbs[StudyAbs.words.str.len() > 0]

Abstracts1.head(2)
Abstracts1.shape

##Abstracts account for 51012
# saving the Abstracts Table (dataframe) 

Abstracts1.to_csv('All_Abstracts.csv') 
#looking through the abstracts for specific terms 

## 

##TABLE OF abstracts related to DIAGNOSTICS 

Diagnostics=Abstracts1[Abstracts1['abstract'].str.contains('diagnostics')]

Diagnostics.shape

## I identify here 651 entries among Abstracts related to DIAGNOSTICS
# saving the Table (dataframe): Study Abstracts related to DIAGNOSTICS

Diagnostics.to_csv('Study_Abstracts_on_Diagnostics.csv') 
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 
#to omit:

symbols1='!@#$%&*.,?"-'

ignoreThese=['background', 'abstract',

             'our','this','the',

             'objective','since', 'name']



for char in symbols1:

        words1=Diagnostics['words'].replace(char,' ')

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
#Abtracts with the term corona

corona=Diagnostics[Diagnostics['abstract'].str.contains('corona')]

corona

## Shows 126 abstracts related to diagnostics and corona
#to omit symbols and terms:

## already defined symbols1='!@#$%&*.,?"-'

## already defined ignoreThese



for char in symbols1:

        words2=corona['words'].replace(char,' ')

#lower case all words

words2=str(words2)

words2=words2.lower()



#ignore words

for item in ignoreThese:

        words2=words2.replace(item, ' ') 

##CLOUD

wordcloud2 = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words2)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud2, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# saving the Table (dataframe): Study Abstracts related to DIAGNOSTICS

corona.to_csv('Study_Abstracts_on_Corona_Diagnostics.csv') 
##Search through Abstracts with the word CORONA for TOOLS

tools=corona[corona['abstract'].str.contains('diagnostic tools')]

tools

##4 results
# saving the Table (dataframe): 

#Four study Abstracts related to DIAGNOSTICS tools

tools.to_csv('Study_Abstracts_on_corona_diagnostics_tools.csv') 
testing=Diagnostics[Diagnostics['abstract'].str.contains('test')]

testing
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS test

testing.to_csv('Study_Abstracts_on_Diagnostics_Test.csv')
#to omit:

## already defined symbols1 above

##update to words to ignore

ignoreThese2=['background', 'abstract','our','this', 'words'

             'the','objective','since','summary',

             'commentary']



for char in symbols1:

        words3=testing['words'].replace(char,' ')

#lower case all words

words3=str(words3)

words3=words3.lower()



#ignore words

for item in ignoreThese2:

        words3=words3.replace(item, ' ') 

##CLOUD

wordcloud3 = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words3)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud3, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
COVID=Diagnostics[Diagnostics['abstract'].str.contains('COVID')]

COVID.shape
COVID
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and COVID

COVID.to_csv('Study_Abstracts_on_Diagnostics_and_COVID.csv')
###

##

sample=Diagnostics[Diagnostics['abstract'].str.contains('sample')]

sample.shape
FDA=Diagnostics[Diagnostics['abstract'].str.contains('FDA')]

FDA

#8 abstracts related to FDA and diagnostics
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and FDA

FDA.to_csv('Study_Abstracts_on_Diagnostics_and_FDA.csv')
#Diagnostics studies show..

studiesD=Diagnostics[Diagnostics['abstract'].str.contains('studies show')]

studiesD

#3 abstracts
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and evidence

studiesD.to_csv('Study_Abstracts_on_what_Diagnostics_studies_show.csv')
evidence=Diagnostics[Diagnostics['abstract'].str.contains('evidence')]

evidence.shape
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and evidence

evidence.to_csv('Study_Abstracts_on_Diagnostics_and_evidence.csv')
ViralTest=Diagnostics[Diagnostics['abstract'].str.contains('viral test')]

ViralTest

#4 abstracts
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and Viral Test

ViralTest.to_csv('Study_Abstracts_on_Diagnostics_and_ViralTest.csv')
RtPCR=Diagnostics[Diagnostics['abstract'].str.contains('RT-PCR')]

RtPCR.shape
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and RtPCR

RtPCR.to_csv('Study_Abstracts_on_Diagnostics_and_RtPCR.csv')
##antibodies tests

Antibodies=Diagnostics[Diagnostics['abstract'].str.contains('antibodies')]

Antibodies.shape
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and Antibodies

Antibodies.to_csv('Study_Abstracts_on_Diagnostics_and_Antibodies.csv')
Sero=Diagnostics[Diagnostics['abstract'].str.contains('serology')]

Sero.shape
# saving the Table (dataframe): 

#Study Abstracts related to DIAGNOSTICS and Serology

Sero.to_csv('Study_Abstracts_on_Diagnostics_and_Serology.csv')
##Assays

assay=Diagnostics[Diagnostics['abstract'].str.contains('assay')]

assay.shape
#to omit:

## already defined symbols1 above

##update words to ignore:

ignoreThese3=['background', 'abstract','our','this', 'words'

             'the','objective','since','summary','name',

              'commentary','study', 'object', 'dtypes',

             'dtype']



for char in symbols1:

        words4=assay['words'].replace(char,' ')

#lower case all words

words4=str(words4)

words4=words4.lower()



#ignore words

for item in ignoreThese3:

        words4=words4.replace(item, ' ') 

##CLOUD

wordcloud4 = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words4)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud4, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
##Other diganostic tools:

#Scans (ct scans)

#Symptoms

#X-rays

scans=Diagnostics[Diagnostics['abstract'].str.contains('scan')]

scans.shape
sympt=Diagnostics[Diagnostics['abstract'].str.contains('symptoms')]

sympt.shape
xrays=Diagnostics[Diagnostics['abstract'].str.contains('X-ray')]

xrays.shape
# saving the Table (dataframe): 

#Study Abstracts related to other DIAGNOSTIC tools

OtherDiag= pd.concat([scans,sympt,xrays])

OtherDiag.to_csv('Study_Abstracts_on_Diagnostics_other_tools.csv')
##Study abstracts related to development in diagnostics

Dev=Diagnostics[Diagnostics['abstract'].str.contains('development')]

Dev.shape
#to omit:

## already defined symbols1 above

##update words to ignore:

ignoreThese4=['background', 'abstract','our','this', 'words',

             'the','objective','since','summary','name',

              'commentary','study', 'object', 'dtypes',

             'dtype','many']



for char in symbols1:

        words5=Dev['words'].replace(char,' ')

#lower case all words

words5=str(words5)

words5=words5.lower()



#ignore words

for item in ignoreThese4:

        words5=words5.replace(item, ' ') 

##CLOUD

wordcloud5 = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words5)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud5, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# saving the Table (dataframe): 

#Study abstracts related to development in diagnostics

Dev.to_csv('Study_abstracts_on_diagnosticsADVANCES_development.csv')
Adv=Diagnostics[Diagnostics['abstract'].str.contains('advances')]

Adv.shape
# saving the Table (dataframe): 

#Study abstracts related to advances in diagnostics

Adv.to_csv('Study_abstracts_on_diagnosticsADVANCES.csv')
Discover=Diagnostics[Diagnostics['abstract'].str.contains('discover')]

Discover.shape
# saving the Table (dataframe): 

#Study abstracts related to diagnostics advances: Discover

Discover.to_csv('Study_abstracts_on_diagnosticsADVANCES_discover.csv')
Future=Diagnostics[Diagnostics['abstract'].str.contains('future')]

Future.shape
# saving the Table (dataframe): 

#Study abstracts related to diagnostics Future

Future.to_csv('Study_abstracts_on_diagnostics_Future.csv')
findings=Diagnostics[Diagnostics['abstract'].str.contains('findings')]

findings.shape
# saving the Table (dataframe): 

#Study abstracts related to diagnostics advances: findings

findings.to_csv('Study_abstracts_on_diagnosticsADVANCES_findings.csv')
#to omit:

## already defined symbols1 above

##update words to ignore:

ignoreThese5=['background', 'abstract','our','this', 'words',

             'the','objective','since','summary','name',

              'commentary','study', 'object', 'dtypes',

             'dtype','many', 'goal','web']



for char in symbols1:

        words6=Dev['words'].replace(char,' ')

#lower case all words

words6=str(words6)

words6=words6.lower()



#ignore words

for item in ignoreThese5:

        words6=words6.replace(item, ' ') 

##CLOUD

wordcloud6 = WordCloud(

            width = 1000,

            height = 1000,

            background_color = 'black',

            stopwords = STOPWORDS).generate(words6)

fig = plt.figure(

    figsize = (20, 10),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud6, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()