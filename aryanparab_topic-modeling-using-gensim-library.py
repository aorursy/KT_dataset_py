# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gensim

from gensim import corpora

from pprint import pprint
documents = ["The Saudis are preparing a report that will acknowledge that", 

             "Saudi journalist Jamal Khashoggi's death was the result of an", 

             "interrogation that went wrong, one that was intended to lead", 

             "to his abduction from Turkey, according to two sources."]



documents_2 = ["One source says the report will likely conclude that", 

                "the operation was carried out without clearance and", 

                "transparency and that those involved will be held", 

                "responsible. One of the sources acknowledged that the", 

                "report is still being prepared and cautioned that", 

                "things could change."]
# Tokenize(split) the sentences into words

texts=[[text for text in doc.split()] for doc in documents]

print(texts)

texts[0]

# Create dictionary

dictionary=corpora.Dictionary(texts)



print(dictionary)
# Show the word to id map

print(dictionary.token2id)
#If you get new documents in the future, it is also possible to update an existing dictionary to include the new words.



texts=[[line for line in doc.split()] for doc in documents_2]  #documents_2 was made where documents was created



dictionary.add_documents(texts)
from gensim.utils import simple_preprocess
#Repeating the above steps

# Tokenize the docs

tokenized_list = [simple_preprocess(doc) for doc in documents]



# Create the Corpus

mydict = corpora.Dictionary()

corpus=[mydict.doc2bow(doc , allow_update = True) for doc in tokenized_list ]

pprint(corpus)

# Show the Word Weights in Corpus

for doc in corpus:



    print([[mydict[id], freq] for id, freq in doc])
from gensim import models

import numpy as np
# Create the TF-IDF model



tfidf=models.TfidfModel(corpus , smartirs='ntc')
# Show the TF-IDF weights

for doc in tfidf[corpus]:

    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
from gensim.models import LdaModel , LdaMulticore

from gensim.utils import simple_preprocess

import nltk 

from nltk import WordNetLemmatizer

from nltk.corpus import stopwords

import re

import string

#Creating a list of stopwords and punctuations. These are in-built functions of their respective libraries.

#You can make your own as well

stop_words = stopwords.words('english') +['the','a','an']

punc = string.punctuation
documents = ['''The UEFA Champions League (abbreviated as UCL, also known as the European Cup) is an annual club football competition organised by the Union of European Football Associations (UEFA) and contested by top-division European clubs, deciding the competition winners through a group and knockout format. It is one of the most prestigious football tournaments in the world and the most prestigious club competition in European football, played by the national league champions (and, for some nations, one or more runners-up) of their national associations.

In its present format, the Champions League begins in late June with a preliminary round, three qualifying rounds and a play-off round, all played over two legs. The six surviving teams enter the group stage, joining 26 teams qualified in advance. The 32 teams are drawn into eight groups of four teams and play each other in a double round-robin system. The eight group winners and eight runners-up proceed to the knockout phase that culminates with the final match in late May or early June.[5] The competition has been won by 22 clubs, 12 of which have won it more than once.[8] Real Madrid is the most successful club in the tournament's history, having won it 13 times, including its first five seasons. Liverpool are the reigning champions, having beaten Tottenham Hotspur 2–0 in the 2019 final. Spanish clubs have the highest number of victories (18 wins), followed by England (13 wins) and Italy (12 wins). ''',

             

'''Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so.

    It involves computers learning from data provided so that they carry out certain tasks. For simple tasks assigned to computers,

    it is possible to program algorithms telling the machine how to execute all steps required to solve the problem at hand; 

on the computer's part, no learning is needed. For more advanced tasks, it can be challenging for a human to manually create the needed algorithms. In practice, it can turn out to be more effective to help the machine develop its own algorithm, rather than having human programmers specify every needed step.Types of supervised learning algorithms include Active learning, classification and regression.[30] Classification algorithms are used when the outputs are restricted to a limited set of values, and regression algorithms are used when the outputs may have any numerical value within a range. Manifold learning algorithms attempt to do so under the constraint that the learned representation is low-dimensional. Sparse coding algorithms attempt to do so under the constraint that the learned representation is sparse, meaning that the mathematical model has many zeros. 

Multilinear subspace learning algorithms aim to learn low-dimensional representations directly from tensor representations for multidimensional data, without reshaping them into higher-dimensional vectors.[42] Deep learning algorithms discover multiple levels of representation, or a hierarchy of features, with higher-level, more abstract features defined in terms of (or generating) lower-level features. It has been argued that an intelligent machine is one that learns a representation that disentangles the underlying factors of variation that explain the observed data.[43]''', 

             

             

             

             '''India Coronavirus Cases: Pune has now overtaken Mumbai as the city with the maximum number of novel Coronavirus infections in Maharashtra. In fact, after Delhi, it has the highest number of cases in the entire country

Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.

The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. 

With 941 casualties in the past 24 hours, the toll in India topped 50,000 (50,921) on Monday. As many as 57,982 new cases were reported in the country, taking the tally to 26,47,664. Of the 26 lakh cases, at least 19 lakh have recovered, while over 6.7 lakh are still active. India has been reporting over 60,000 cases daily since August 7, barring August 11 when the country registered 53,601 new instances of the infection.

       At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments. WHO will continue to provide updated information as soon as clinical findings become available.Coronavirus India News Live Updates: The Bihar government Monday extended the lockdown in its state till September 6 in view of the prevailing COVID-19 situation, According to a notification issued by the Home Department restrictions will remain in place in the district headquarters,

       sub-divisional headquarters, block headquarters and all municipal areas.         

             ''']

from nltk.corpus import wordnet



def get_wordnet_pos(word):

    

    

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN) 
data_processed=[]

for line in documents:

    lemmatizer=WordNetLemmatizer()

    doc_out=[]

    for word in line.split() : 

        word=''.join([w.lower() for w in word if w not in ['[',']','(',')',',','.']]) #Removing [] from words like [2] etc as seen in wiki documents

        if word not in stop_words and word not in punc:

            lemmatized_wd= lemmatizer.lemmatize(word, get_wordnet_pos(word))#lemmatizing the words with its part of speech tag

            if lemmatized_wd:

                 doc_out.append(lemmatized_wd)

        else : 

            continue

    data_processed.append(doc_out)

print(data_processed)
new_dict=corpora.Dictionary(data_processed)

corpus=[new_dict.doc2bow(line) for line in data_processed]

print(corpus)
lda_model= LdaMulticore(corpus=corpus,id2word=new_dict,random_state=100,num_topics=3,

                         alpha='asymmetric',

                         decay=0.5,

                         iterations=200)



lda_model.print_topics(-1)
from gensim.models import LsiModel
lsi_model = LsiModel(corpus=corpus, id2word=new_dict, num_topics=3, decay=0.5)
pprint(lsi_model.print_topics(-1))
document_test=[

    '''The top-ranked UEFA competition is the UEFA Champions League, which started in the 1992/93 season and gathers the top 1–4 teams of each country's league (the number of teams depend on that country's ranking and can be upgraded or downgraded); this competition was re-structured from a previous one that only gathered the top team of each country (held from 1955 to 1992 and known as the European Champion Clubs' Cup or simply the European Cup).



A second, lower-ranked competition is the UEFA Europa League. This competition, for national knockout cup winners and high-placed league teams, was launched by UEFA in 1971 as a successor of both the former UEFA Cup and the Inter-Cities Fairs Cup (also begun in 1955). A third competition, the UEFA Cup Winners' Cup, which had started in 1960, was absorbed into the UEFA Cup (now UEFA Europa League) in 1999.



In December 2018, UEFA announced the creation of a third club competition, with a working title of Europa League 2 (UEL2) (The name was later decided as UEFA Europa Conference League) . The competition would feature 32 teams directly in 8 groups of 4, with a knockout round between the second placed teams in UEFA Europa Conference League and the third placed teams in the Europa League, leading to a final 16 knockout stage featuring the eight group winners. UEFA announced that the first edition of the competition begins in 2021 [12].''',

    

    '''In March, after the lockdown was imposed, the United Nations (UN) and the World Health Organization (WHO) praised 

    India's response to the pandemic as 'comprehensive and robust,' terming the lockdown restrictions as 'aggressive but vital' 

    for containing the spread and building necessary healthcare infrastructure. At the same time, the Oxford COVID-19 Government Response

    Tracker (OxCGRT) noted the government's swift and stringent actions, emergency policy-making, emergency investment in health care, 

    fiscal stimulus, investment in vaccine and drug R&D and gave India a score of 100 for the strict response.

    Also in March, Michael Ryan, chief executive director of the WHO's health emergencies programme noted that India had tremendous 

    \capacity to deal with the outbreak owing to its vast experience in eradicating smallpox and polio.[20][21][22] Other commentators

    have raised concerns about the economic fallout arising as a result of the pandemic and preventive restrictions.[23][24] The lockdown was justified by the government and other agencies for being preemptive to prevent India from entering a higher stage which could make handling very difficult and cause even more losses thereafter.[25][26]'''

]
data_processed=[]

for line in document_test:

    lemmatizer=WordNetLemmatizer()

    doc_out=[]

    for word in line.split() : 

        word=''.join([w.lower() for w in word if w not in ['[',']','(',')',',','.']])

        if word not in stop_words and word not in punc:

            lemmatized_wd= lemmatizer.lemmatize(word, get_wordnet_pos(word))

            if lemmatized_wd:

                 doc_out.append(lemmatized_wd)

        else : 

            continue

    data_processed.append(doc_out)



corpus_test=[new_dict.doc2bow(line) for line in data_processed]
top_topics = lda_model.get_document_topics(corpus_test, minimum_probability=0.0)
for i in top_topics:

    print(i)

    max=-1

    topic=1

    for topic_no , predict in i:

        if predict>max:

            max=predict

            topic=topic_no

    print(topic,max)

            