import matplotlib.pyplot as plt

import nltk

import numpy as np

import operator

import pandas as pd

import seaborn as sns



from collections import Counter

from nltk.tokenize import PunktSentenceTokenizer

from nltk.chunk import RegexpParser
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding="ISO-8859-1", low_memory=False)

data.head(n=5)
freq_dist = data['country_txt'].value_counts()

freq_dist[freq_dist > 2000]
iraq_data = data[ data['country_txt'] == 'Iraq' ]

iraq_data['gname'].value_counts()[:10]
isis_iraq_data = iraq_data[iraq_data['gname'] == 'Islamic State of Iraq and the Levant (ISIL)']
data = isis_iraq_data['iyear'].value_counts()

data
for year in range(2013,2017):

    num_kills = sum(isis_iraq_data['nkill'][isis_iraq_data['iyear'] == year].dropna())

    print("Number of kills by ISIL in Iraq in ", year, " = ", num_kills)
isis_iraq_data['attacktype1_txt'].value_counts()
isis_iraq_data[['success' ,'nkill', 'provstate']][isis_iraq_data['attacktype1_txt'] == 'Hostage Taking (Barricade Incident)']
barricade = isis_iraq_data['nkill'][isis_iraq_data['attacktype1_txt'] == 'Hostage Taking (Barricade Incident)']

sns.distplot(barricade, label="Barricade")



plt.legend()

plt.show()
isis_iraq_data['targtype1_txt'][isis_iraq_data['attacktype1_txt'] == 'Hostage Taking (Kidnapping)'].value_counts()
isis_iraq_data['targsubtype1_txt'][(isis_iraq_data['targtype1_txt'] == 'Private Citizens & Property') & (isis_iraq_data['attacktype1_txt'] == 'Hostage Taking (Kidnapping)') ].value_counts()
isis_iraq_data['target1'][ (isis_iraq_data['targsubtype1_txt'] == 'Laborer (General)/Occupation Identified') & (isis_iraq_data['targtype1_txt'] == 'Private Citizens & Property') & (isis_iraq_data['attacktype1_txt'] == 'Hostage Taking (Kidnapping)') ].value_counts()
isis_iraq_data['attacktype1_txt'][ (isis_iraq_data['targtype1_txt'] == 'Terrorists/Non-State Militia')].value_counts()
targets = isis_iraq_data[['target1']][ (isis_iraq_data['attacktype1_txt'] == 'Assassination') & (isis_iraq_data['targtype1_txt'] == 'Terrorists/Non-State Militia') ]

targets.apply( lambda x: x.str.split(':')[0][0] , axis=1).value_counts()
def text_manipulation(text, regex):

    words = nltk.word_tokenize(text.lower())

    tagged = nltk.pos_tag(words)



    chunkGram = regex

    

    chunkParser = RegexpParser(chunkGram)

    chunked = chunkParser.parse(tagged)



    candidate_keywords = []

    for tree in chunked.subtrees():

        if (tree.label() == 'PHRASE') and (len(tree.leaves()) >= 2):

            candidate_keyword = ' '.join([x for x,y in tree.leaves()])

            candidate_keywords.append(candidate_keyword)



    return Counter(candidate_keywords)

    

text = ' '.join(isis_iraq_data['summary'][isis_iraq_data['targtype1_txt'] == 'Terrorists/Non-State Militia'].dropna().tolist())

common_phrases = text_manipulation(text, r""" PHRASE: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}""")

for x,y in sorted(common_phrases.items(), key=operator.itemgetter(1), reverse=True):

    if y>=2:

        print((x,y))
text = "The big bad wolf killed the lazy fox"

words = nltk.word_tokenize(text)

nltk.pos_tag(words)
text = ' '.join(isis_iraq_data['summary'][isis_iraq_data['targtype1_txt'] == 'Terrorists/Non-State Militia'].dropna().tolist())

common_phrases = text_manipulation(text, r""" PHRASE: {<JJ>? <NN.*>+ <VB.*>+ <DT>? <JJ>? <NN.*>+ }""")

for x,y in sorted(common_phrases.items(), key=operator.itemgetter(1), reverse=True):

    if y>=2:

        print((x,y))
text = ' '.join(isis_iraq_data['motive'].dropna().tolist())

common_phrases = text_manipulation(text,  r""" PHRASE: {<VB> <DT>? <NN.*>+ (<CC> <NN*>+)? <IN>? <NN.*>+}""")

for x,y in sorted(common_phrases.items(), key=operator.itemgetter(1), reverse=True):

    if y>=1:

        print((x,y))
isis_iraq_data['motive'].dropna().value_counts()[:10]
isis_iraq_data[['target1', 'success','attacktype1_txt']][isis_iraq_data['motive'].str.contains('Baghdad', regex=True, na=False)]
data = dict(isis_iraq_data['provstate'].value_counts())

x = list(data.keys())

y = list(data.values())

d = pd.DataFrame({'Place':x, 'Num Cases':y})



# Make the plot bigger

a4_dims = (13, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



sns.barplot(data=d[:10], x='Place', y='Num Cases', ax=ax) #Display top 10

plt.show()
data = pd.DataFrame(columns=['Place', 'Num Kills'])



i=0

for place in isis_iraq_data['provstate'].unique():

    num_kills = 0

    

    for year in range(2013,2017):

        num_kills += sum(isis_iraq_data['nkill'][(isis_iraq_data['iyear'] == year) & (isis_iraq_data['provstate'] == place)].dropna())

        

    data.loc[i] = [place, num_kills]

    i+=1

        

data = data.sort_values(by='Num Kills', ascending=False) 



# Make the plot bigger

a4_dims = (13, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



sns.barplot(data=data[:10], x='Place', y='Num Kills', ax=ax) #Display top 10

plt.show()
data = pd.DataFrame(columns=['Place', '2013', '2014', '2015', '2016', 'Total Kills'])



i=0

for place in isis_iraq_data['provstate'].unique():

    num_kills = []

    

    for year in range(2013,2017):

        num_kills.append(sum(isis_iraq_data['nkill'][(isis_iraq_data['iyear'] == year) & (isis_iraq_data['provstate'] == place)].dropna()))

        

    data.loc[i] = [place, num_kills[0], num_kills[1], num_kills[2], num_kills[3], sum(num_kills)]

    i+=1

        

data = data.sort_values(by='Total Kills', ascending=False) 

data
df = pd.melt(data, id_vars="Place", var_name="Year", value_name="Kill Count")

places = df[ (~df['Place'].isin(['Wasit', 'Basra', 'Dhi Qar', 'Najaf', 'Maysan', 'Muthanna','Kirkuk', 

                                 'Al Qadisiyah', 'NIneveh', 'Sulaymaniyah','Unknown', 'Dihok'])) & 

             (~df['Year'].str.contains('Total Kills'))] #Exclude certain places



df[:30] #Just the first 30 entries for understanding
# Make the plot bigger

a4_dims = (13, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



sns.factorplot(x='Place', y='Kill Count', hue='Year', data=places, kind='bar', ax=ax)

plt.show()