import numpy as np

import matplotlib.pyplot as plt

import datetime

import calendar

import pandas as pd 

import os

from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS

import networkx as nx

print(os.listdir("../input"))
data=pd.read_excel('../input/global_terrorism_database.xlsx')

data
decade_north_america=[]

decade_total=[]

decade_labels=[1970,1980,1990,2000,2010]

for i in range(len(data['region_txt'])):

    crime_year=int(data['iyear'][i])

    if data['region_txt'][i]=="North America":

        decade_north_america.append(crime_year)

    decade_total.append(crime_year)

plt.hist(decade_north_america,bins=[1970,1980,1990,2000,2010,2020])

plt.xlabel("Year")

plt.ylabel("Frequency of terrorist attacks in North America")

plt.show()
plt.hist(decade_total,bins=[1970,1980,1990,2000,2010,2020])

plt.xlabel("Year")

plt.ylabel("Frequency of terrorist attacks")

plt.show()
crime_continent={}

for i in range(len(data['region_txt'])):

    crime_year=int(data['iyear'][i])

    if crime_year >= 2010  and crime_year < 2020:

        if data['region_txt'][i] not in crime_continent:

            crime_continent[data['region_txt'][i]]=0

        crime_continent[data['region_txt'][i]]+=1

plt.rcParams["figure.figsize"]=[35,8]

plt.bar(*zip(*crime_continent.items()))

plt.xlabel("Continent")

plt.ylabel("Frequency of terrorist attacks in 2010-2020")

plt.show()
stopwords = set(STOPWORDS)

total_words = []

for index, row in data.iterrows():

    summary = row['summary']

    try:

        curr_word = ' '.join(summary.split(":")[1:])

        total_words += curr_word.split()

    except:

        pass

total_words = ' '.join(total_words)



wordcloud = WordCloud(width=480, height=480, 

            stopwords=stopwords,colormap="Oranges_r").generate(total_words) 



plt.figure() 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0) 

plt.show()
total_words=[]

for index, row in data.iterrows():

    motive = row['motive']

    try:

        total_words += motive.split()

    except:

        pass

total_words = ' '.join(total_words)

wordcloud = WordCloud(width=480, height=480, 

            stopwords=stopwords,).generate(total_words) 

plt.figure(figsize=(20,20)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis("off") 

plt.margins(x=0, y=0) 

plt.show()
G=nx.DiGraph()

for index, row in data.iterrows():

    source=int(row['eventid'])

    related=row['related']

    if type(related) !=float:

        G.add_node(source)

        try: 

            related=related.split(' ')

            for edge in range(len(related)):

                related[edge]=int(related[edge].strip(','))

                G.add_node(related[edge])

                G.add_edge(source,related[edge])

        except:

            related=[]

        try:

            related=related.split(',')

            for edge in range(len(related)):

                related[edge]=int(related[edge].strip(' '))

                G.add_node(related[edge])

                G.add_edge(source,related[edge])

        except:

            related=[]

scc=list(nx.strongly_connected_components(G))

print("Total no of coordianted Terror attack = " + str(len(scc)))

largest_set=-100

for terror_set in scc:

    largest_set=max(largest_set,len(terror_set))

print("Largest set of coordinated Terror attack = "+ str(largest_set))