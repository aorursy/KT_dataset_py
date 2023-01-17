!pip install textblob_fr     #for installation

import pandas as pd 

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

import re

import spacy

from spacy.lang.fr.stop_words import STOP_WORDS

import string

from textblob import Blobber

from textblob_fr import PatternTagger, PatternAnalyzer

tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

import plotly.graph_objects as go

import plotly.express as px

data = pd.read_csv('../input/insurance-reviews-france/Comments.csv')

data.head()
data= data.drop(['Unnamed: 0'],axis=1)
NAN = [(c, data[c].isna().mean()*100) for c in data]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN.sort_values("percentage", ascending=False)
data =data.dropna()
NAN = [(c, data[c].isna().mean()*100) for c in data]

NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN.sort_values("percentage", ascending=False)
data["Comment"]= data["Comment"].str.lower()
AComment=[]

for comment in data["Comment"].apply(str):

    Word_Tok = []

    for word in  re.sub("\W"," ",comment ).split():

        Word_Tok.append(word)

    AComment.append(Word_Tok)

data["Word_Tok"]= AComment

data.head()
stop_words=set(STOP_WORDS)



deselect_stop_words = ['n\'', 'ne','pas','plus','personne','aucun','ni','aucune','rien']

for w in deselect_stop_words:

    if w in stop_words:

        stop_words.remove(w)

    else:

        continue
AllfilteredComment=[]

for comment in data["Word_Tok"]:

    filteredComment = [w for w in comment if not ((w in stop_words) or (len(w) == 1))]

    AllfilteredComment.append(' '.join(filteredComment))
data["CommentAferPreproc"]=AllfilteredComment

data.head()
senti_list = []

for i in data["CommentAferPreproc"]:

    vs = tb(i).sentiment[0]

    if (vs > 0):

        senti_list.append('Positive')

    elif (vs < 0):

        senti_list.append('Negative')

    else:

        senti_list.append('Neutral')   

data["sentiment"]=senti_list

data.head()
Number_sentiment= data.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)
fig = px.histogram(data, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment, values=Number_sentiment['Name'], names=Number_sentiment['sentiment'], color_discrete_sequence=px.colors.sequential.Emrld

)

fig.show()
fig = px.histogram(data, x="Name",color="Name")

fig.update_layout(

    title_text='Number of Comments per Assurance', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(data, x="Year",color="Year")

fig.update_layout(

    title_text='Number of Comments per Year', # title of plot

    xaxis_title_text='Year', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(data, x="Month",color="Month")

fig.update_layout(

    title_text='Number of Comments per Month', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(data, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(data, x="Year",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Year', # title of plot

    xaxis_title_text='Year', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(data, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2015 = data [(data['Year'] == 2015) ].reset_index(drop=True)
Number_sentiment_2015= Data_2015.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)
fig = px.histogram(Data_2015, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2015', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2015, values=Number_sentiment_2015['Name'], names=Number_sentiment_2015['sentiment'], color_discrete_sequence=px.colors.sequential.Darkmint

)

fig.show()
fig = px.histogram(Data_2015, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2015', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2015, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2015', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2016 = data [(data['Year'] == 2016) ].reset_index(drop=True)
Number_sentiment_2016= Data_2016.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)
Data_2016 = data [(data['Year'] == 2016) ].reset_index(drop=True)

fig = px.histogram(Data_2016, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2016', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2016, values=Number_sentiment_2016['Name'], names=Number_sentiment_2016['sentiment'], color_discrete_sequence=px.colors.sequential.Darkmint

)

fig.show()
fig = px.histogram(Data_2016, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2016', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2016, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2016', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2017 = data [(data['Year'] == 2017) ].reset_index(drop=True)
Number_sentiment_2017= Data_2017.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)
fig = px.histogram(Data_2017, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2017', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2017, values=Number_sentiment_2017['Name'], names=Number_sentiment_2017['sentiment'], color_discrete_sequence=px.colors.sequential.Emrld

)

fig.show()
fig = px.histogram(Data_2017, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2017', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2017, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2017', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2018 = data [(data['Year'] == 2018) ].reset_index(drop=True)
Number_sentiment_2018= Data_2018.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)

fig = px.histogram(Data_2018, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2018', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2018, values=Number_sentiment_2018['Name'], names=Number_sentiment_2018['sentiment'], color_discrete_sequence=px.colors.sequential.Darkmint

)

fig.show()
fig = px.histogram(Data_2018, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2018', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2018, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2018', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2019 = data [(data['Year'] == 2019) ].reset_index(drop=True)
Number_sentiment_2019= Data_2019.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)
fig = px.histogram(Data_2019, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2019', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2019, values=Number_sentiment_2019['Name'], names=Number_sentiment_2019['sentiment'], color_discrete_sequence=px.colors.sequential.Darkmint

)

fig.show()
fig = px.histogram(Data_2019, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2019', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2019, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2019', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Data_2020 = data [(data['Year'] == 2020) ].reset_index(drop=True)
Number_sentiment_2020= Data_2020.groupby(["sentiment"])["Name"].count().reset_index().reset_index(drop=True)

fig = px.histogram(Data_2020, x="sentiment",color="sentiment")

fig.update_layout(

    title_text='Sentiment of reviews in 2020', # title of plot

    xaxis_title_text='Sentiment', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.pie(Number_sentiment_2020, values=Number_sentiment_2020['Name'], names=Number_sentiment_2020['sentiment'], color_discrete_sequence=px.colors.sequential.Emrld

)

fig.show()
fig = px.histogram(Data_2020, x="Name",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Assurance in 2020', # title of plot

    xaxis_title_text='Assurance', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
fig = px.histogram(Data_2020, x="Month",color="sentiment")

fig.update_layout(

    title_text='Sentiments per Month in 2020', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Number of Comments', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()