import pandas as pd

import spacy

import matplotlib.pyplot as plt

import numpy as np

nlp = spacy.load("en_core_web_sm")

import seaborn as sns

import re
with open("../input/trump-vs-biden-debate-transcript-analysis/Transcript.txt","r",encoding="utf-8") as f:

    text = f.read()
doc = nlp(str(text)) ##Creating a spacy doc object
# print(doc)
# Function to plot the wordcloud

def plot_cloud(wordcloud):

    plt.figure(figsize=(40,30))

    plt.imshow(wordcloud)

    plt.axis('off')
#!pip install wordcloud

from wordcloud import WordCloud , STOPWORDS
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str(text))

plot_cloud(wordcloud)
with open("../input/trump-vs-biden-debate-transcript-analysis/Transcript.txt","r",encoding="utf-8") as f:

    text_lines = f.readlines() 
sentences = []

for line in text_lines:

    sentences.append(line)
df = pd.DataFrame(sentences,columns=["sentence"])

df.head()
import nltk

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df['scores'] = df['sentence'].apply(lambda sent: sid.polarity_scores(sent))

df.head()



# It returns a dictionary with scores
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])



df.head()
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')



df.head()
sns.set(font_scale=2)

sns.countplot(df["comp_score"])
negative_sent = df[df.comp_score=="neg"] #Separate dataframe with negative senences
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str(negative_sent["sentence"]))

plot_cloud(wordcloud)
positive_sent = df[df.comp_score=="pos"] #positive sentences
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str(positive_sent["sentence"]))

plot_cloud(wordcloud)
ORGS =[]

for ent in doc.ents:

    if ent.label_ == "ORG":

        if ent.text not in ORGS:

            ORGS.append(ent.text)
ORGS
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str(ORGS))

plot_cloud(wordcloud)
PERS =[]

for ent in doc.ents:

    if ent.label_ == "PERSON":

        if ent.text not in PERS:

            PERS.append(ent.text)
PERS
re.search("Kamala Harris",text)

text[126400:126530]
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str([PERS]))

plot_cloud(wordcloud)
GPES =[]

for ent in doc.ents:

    if ent.label_ == "GPE":

        if ent.text not in GPES:

            GPES.append(ent.text)
GPES
import re

ind=re.search("India",text)

text[26350:26990]
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color="salmon" , colormap="Pastel1" , collocations=False, stopwords=STOPWORDS).generate(str([GPES]))

plot_cloud(wordcloud)
EVENTS =[]

for ent in doc.ents:

    if ent.label_ == "EVENT":

        if ent.text not in EVENTS:

            EVENTS.append(ent.text)
EVENTS
for ent in doc.ents:

    print(ent.text+" "+ ent.label_ + spacy.explain(str(ent.label_)) )

    print("\n")

    print("\n")