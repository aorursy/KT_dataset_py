from textblob import TextBlob

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

from collections import Counter

sns.set()



with open('../input/lotr.txt', 'r', errors='ignore') as file:

    data = file.read().replace('\n', '')



text= TextBlob(data)

my_list=text.tags
Eriador = data.count("Eriador")

Arnor = data.count("Arnor")

Rohan = data.count("Rohan")

Gondor = data.count("Gondor")

Mordor = data.count("Mordor")

Rhun = data.count("Rhun")

realms_list = [['Gondor', Gondor], ['Mordor', Mordor], ['Rohan', Rohan], ['Arnor', Arnor], ['Eriador', Eriador]]

df_realms=pd.DataFrame(realms_list, columns=['Realm', 'Times mentioned'])

colors = ["crimson", "forrest green", "true blue", "amber", "black"]

sns.set_style("darkgrid")

plt.figure(figsize=(10, 5))

with sns.xkcd_palette(colors):

    sns.barplot(x="Realm", y="Times mentioned", saturation=0.9, data=df_realms).set_title("Lord of the Rings - number of times realms being mentioned")
Orc = data.count("Orc")+data.count("Orcs")+data.count("orc")+data.count("orcs")+data.count("orcish")

Human = data.count("Man")+data.count("Mankind")+data.count("Men")+data.count("men")+data.count("human")

Elf = data.count("Elf")+data.count("Elves")+data.count("elf")+data.count("elves")+data.count("elven")

Dwarf = data.count("Dwarf")+data.count("Dwarves")+data.count("dwarf")+data.count("dwarves")+data.count("dwarven")

Halfling = data.count("Halfling")+data.count("Hobbit")+data.count("Hobbits")+data.count("Halflings")+data.count("halfling")+data.count("hobbit")+data.count("hobbits")+data.count("halflings")

Ent = data.count("Ents")+data.count("Ent")

Troll = data.count("Troll")+data.count("troll")+data.count("Trolls")+data.count("trolls")

Dragon = data.count("dragon")+data.count("dragons")+data.count("Dragon")+data.count("Dragons")

Balrog = data.count("Balrog")+data.count("Balrogs")+data.count("balrog")+data.count("balrogs")

Goblin = data.count("Goblin")+data.count("Goblins")+data.count("goblin")+data.count("goblins")

Warg = data.count("Warg")+data.count("Wargs")+data.count("warg")+data.count("wargs")

Huorn = data.count("Huorn")+data.count("Huorns")+data.count("huorn")+data.count("huorns")

Beorning = data.count("Beorning")+data.count("Beornings")+data.count("beorning")+data.count("beornings")+data.count("Skin-changers")+data.count("Skin-changer")+data.count("skin-changer")+data.count("skin-changers")

races_list = [['Men', Human], ['Hobbits/Halflings', Halfling], ['Elves', Elf], ['Orcs', Orc], ['Dwarves', Dwarf],  ['Goblins', Goblin], ['Ents', Ent], ["Dragons", Dragon], ['Trolls', Troll], ["Wargs", Warg], ['Huorns', Huorn], ["Balrogs", Balrog], ["Beornings", Beorning]]

df_races=pd.DataFrame(races_list, columns=['Race', 'Times mentioned'])

colors = ["amber", "brown", "dark sea green", "forrest green", "crimson", "black",  "brown", "true blue", "black", "forrest green", "brown", "crimson", "blue"]

sns.set_style("darkgrid")

plt.figure(figsize=(15, 7))

with sns.xkcd_palette(colors):

    sns.barplot(x="Race", y="Times mentioned", saturation=0.9, data=df_races).set_title("Lord of the Rings - number of times races being mentioned")
full_t = pd.DataFrame(my_list)

full_t.columns = ['Words', "Word type"]

xft=full_t.groupby('Word type').count().reset_index()

top20ft=xft.nlargest(20, 'Words')



sns.set_style("darkgrid")

plt.figure(figsize=(10, 5))

sns.barplot(x="Words", y="Word type", palette="rocket", saturation=0.9, data=top20ft).set_title("Lord of the Rings - top 20 word types used")
def word_analysis(word_type):

    filtered = [row for row in my_list if str(word_type) in row[1]]

    print("filtered for " + word_type)

    df = pd.DataFrame(filtered)

    df.columns = ["Word", "Occurences"]

    x=df.groupby('Word').count().reset_index()

    y=x.sort_values(by=['Occurences'], ascending=False)

    top10=y.nlargest(10, 'Occurences')

    plt.figure(figsize=(10, 5))

    sns.barplot(x="Word", y="Occurences", palette="rocket", saturation=0.9, data=top10).set_title("Lord of the rings - most frequently used "+ word_type +" type word")
word_type = 'NN'

word_analysis(word_type)
word_type = 'NNP'

word_analysis(word_type)
word_type = 'JJ'

word_analysis(word_type)
word_type = 'VB'

word_analysis(word_type)
sentiment=[]

x=0



for sentence in text:

    text.sentiment



for sentence in text.sentences:

    sentiment.append(sentence.sentiment)

    

sentence_df = pd.DataFrame(sentiment)

sentence_df.describe()
sentence_df['order'] = sentence_df.index

sentence_df = pd.DataFrame(sentiment)

sentence_df['order'] = sentence_df.index

polarity = pd.Series(sentence_df["polarity"])

plt.figure(figsize=(10, 10))

sns.jointplot("order", "polarity", data=sentence_df, kind="kde")
plt.figure(figsize=(15, 5))

sns.jointplot("order", "polarity", data=sentence_df[sentence_df.polarity != 0], kind="kde")
subjectivity = pd.Series(sentence_df["subjectivity"])

plt.figure(figsize=(10, 10))

sns.jointplot("order", "subjectivity", data=sentence_df, kind="kde")
sns.jointplot("order", "subjectivity", data=sentence_df[sentence_df.subjectivity != 0], kind="kde")
plt.figure(figsize=(10, 10))

sns.jointplot("polarity", "subjectivity", data=sentence_df[(sentence_df.subjectivity != 0)], kind="kde")