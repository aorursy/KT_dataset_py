#-------For DataFrame and Series data manipulation

import pandas as pd

import numpy as np

#-------Data visualisation imports

import seaborn as sns

import matplotlib.pyplot as plt

#-------Interactive data visualisation imports

from plotly import __version__

#print(__version__)

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()

#-------To make data visualisations display in Jupyter Notebooks

%matplotlib inline

#-------To split data into Training and Test Data

from sklearn.model_selection import train_test_split

#-------To make pipelines

from sklearn.pipeline import Pipeline

#-------For Natural Language Processing data cleaning

from sklearn.feature_extraction.text import TfidfTransformer

#CountVectorizer converts collection of text docs to a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer

import string

import nltk

from nltk.corpus import stopwords

#-------For model scoring

from sklearn.metrics import classification_report
trainmessages = pd.read_csv('../input/train.tsv', sep='\t')
trainmessages.info()
trainmessages['Phrase'].describe()
trainmessages.head()
#we have to import the test data to fit to our model

testmessages = pd.read_csv('../input/test.tsv', sep='\t')

testmessages.head()
sns.countplot(data=trainmessages,x='Sentiment')
#to get the numerical values of the above countplot

trainmessages['Sentiment'].iplot(kind='hist')
trainmessages.isnull().sum()
trainmessages.isna().sum()
trainmessages['Length'] = trainmessages['Phrase'].apply(lambda x: len(str(x).split(' ')))
trainmessages['Length'].unique()
data = [dict(

  type = 'box',

  x = trainmessages['Sentiment'],

  y = trainmessages['Length'],

  transforms = [dict(

    type = 'groupby',

    groups = trainmessages['Sentiment'],

  )]

)]

iplot({'data': data}, validate=False)
sns.pairplot(trainmessages,hue='Sentiment',vars=['PhraseId','SentenceId','Length'])
#double-check for any empty Phrases

trainmessages = trainmessages[trainmessages['Phrase'].str.len() >0]
#are there any empty Phrases? if so let's remove them

trainmessages[trainmessages['Phrase'].str.len() == 0].head()
trainmessages = trainmessages[trainmessages['Phrase'].str.len() != 0]
trainmessages[trainmessages['Phrase'].str.len() == 0].head()
#create function to clean data

def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all capitalized words

    2. Remove all punctuation

    3. Remove all stopwords

    4. Returns a list of the cleaned text

    """

    #Remove capitalized words (movie names, actor names, etc.)

    nocaps = [name for name in mess if name.islower()]

    

    #Join the characters again to form the string.

    nocaps = ' '.join(nocaps)

    

    # Check characters to see if they are in punctuation

    nopunc = [char for char in nocaps if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    nostopwords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    

    # Join the characters again to form the string.

    nostopwords = ' '.join(nostopwords)

    

    return nostopwords
#because of our imbalanced classes (categorical labels) we can try over-sampling (making copies of the under-represented classes)

sent_2 = trainmessages[trainmessages['Sentiment']==2]

#we will copy class 0 11 times

sent_0 = trainmessages[trainmessages['Sentiment']==0]

#we will copy class 1 2 times

sent_1 = trainmessages[trainmessages['Sentiment']==1]

#we will copy class 3 2 times

sent_3 = trainmessages[trainmessages['Sentiment']==3]

#we will copy class 4 8 times

sent_4 = trainmessages[trainmessages['Sentiment']==4]



#-----------------------------------------------------

trainmessages = trainmessages.append([sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0,sent_0])

trainmessages = trainmessages.append([sent_1,sent_1])

trainmessages = trainmessages.append([sent_3])

trainmessages = trainmessages.append([sent_4,sent_4,sent_4,sent_4,sent_4,sent_4,sent_4])
#to check the amounts of each class

sns.countplot(data=trainmessages,x='Sentiment')
#we split our train.tsv into training and test data to test model performance

X = trainmessages['Phrase']

y = trainmessages['Sentiment']

msg_train,msg_test,label_train,label_test = train_test_split(X,y)
#let's try using the RandomForestClassifier model to predict

from sklearn.ensemble import RandomForestClassifier

pipelineRFC = Pipeline([

    ('bow',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',RandomForestClassifier())

])
pipelineRFC.fit(msg_train,label_train)

preds = pipelineRFC.predict(msg_test)

print(classification_report(label_test,preds))
#we choose the pipeline with the BEST most ACCURATE model and store the predictions in a variable

preds = pipelineRFC.predict(testmessages['Phrase'])
sub = pd.DataFrame(columns=['PhraseId','Sentiment'])

sub['PhraseId'] = testmessages['PhraseId']

sub['Sentiment'] = pd.Series(preds)
sub.head()
sns.countplot(data=sub,x='Sentiment')
#Convert DataFrame to a csv file that can be uploaded

subfile = 'RT Movie Review Predictions.csv'

sub.to_csv(subfile,index=False)

print('Saved file: ' + subfile)