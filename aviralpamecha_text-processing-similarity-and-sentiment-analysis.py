import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
import spacy
ep_desc = pd.read_csv('/kaggle/input/chai-time-data-science/Description.csv')
ep_desc.head()
ep1_sub = pd.read_csv('/kaggle/input/chai-time-data-science/Cleaned Subtitles/E1.csv')
ep1_sub.head(10)
nlp = spacy.load('en_core_web_lg')
ep1_sub['Text'][0]
doc = nlp(u'Hey, this is Sanyam Bhutani and you\'re listening to "Chai Time Data Science", a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things about data science.\n\nSanyam Bhutani  0:46  \nHello, and welcome to the first episode. This is also part 26 of interview with machine learning heroes. In this episode, I\'m honored to be joined by Abhishek Thakur, chief data scientist at boost.ai, and also the world\'s first, and at the time of recording, only triple Grand Master on Kaggle. Abhishek has been working as a data scientist for the past few years. He also has a background in computer science with a master\'s degree from the University of Bonn. We talked about his journey into data science, his Kaggle experience and his current projects. Enjoy the show.\n\nSanyam Bhutani  1:38  \nHi, everyone. Thanks for tuning in. I\'m really honored to be talking to the world\'s first triple Grand Master today. Thank you so much for taking the time to do this interview Abhishek.')
for token in doc:

    print(token)
doc1 = nlp(u'Hey, this is Sanyam Bhutani and you\'re listening to "Chai Time Data Science", a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things about data science.\n\nSanyam Bhutani  0:46  \nHello, and welcome to the first episode.')



for token in doc1:

    print(token.text, end=' | ')



print('\n----')



for ent in doc1.ents:

    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
doc2 = nlp(u'Hey, this is Sanyam Bhutani and you\'re listening to "Chai Time Data Science", a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things about data science.\n\nSanyam Bhutani  0:46  \nHello, and welcome to the first episode.')



for chunk in doc2.noun_chunks:

    print(chunk.text)
doc1 = nlp(u'Hey, this is Sanyam Bhutani and you\'re listening to "Chai Time Data Science", a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things about data science.\n\nSanyam Bhutani  0:46  \nHello, and welcome to the first episode.')



for token in doc1:

    print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
doc1 = nlp(u'Hey, this is Sanyam Bhutani and you\'re listening to "Chai Time Data Science", a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things about data science.\n\nSanyam Bhutani  0:46  \nHello, and welcome to the first episode.')
doc1.vector
doc2 = nlp(u'Enjoy the show.\n\nSanyam Bhutani  1:38  \nHi, everyone. Thanks for tuning in. I\'m really honored to be talking to the world\'s first triple Grand Master today. Thank you so much for taking the time to do this interview Abhishek.')
doc2.vector
#LEST SEE THAT HOW MUCH HOST AND GUEST WERE SAYING SIMILAR THINGS

for i in range(225):

    

    doc1 = nlp(ep1_sub['Text'][i])

    doc2 = nlp(ep1_sub['Text'][i+1])

    

    print(doc1.similarity(doc2))

    

    

import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer



sid = SentimentIntensityAnalyzer()
doc1 = ep1_sub['Text'][0]
sid.polarity_scores(doc1)

#The starting of the podcast is very much neutral. 
for i in ep1_sub['Text']:

    a = sid.polarity_scores(i)

    print(a)



#These are the Sentiment Scores of every Text of Episode one     



    
ep1_sub.head()
ep1_sub['scores'] = ep1_sub['Text'].apply(lambda review: sid.polarity_scores(review))



ep1_sub.head()
ep1_sub['compound']  = ep1_sub['scores'].apply(lambda score_dict: score_dict['compound'])

ep1_sub.head()
ep1_sub['comp_score'] = ep1_sub['compound'].apply(lambda c: 'positive' if c >=0 else 'negative')



ep1_sub.head()
sns.countplot(x = ep1_sub['comp_score'] )  



#SO THERE ARE VERY FEW NEGATIVE STATEMENTS.