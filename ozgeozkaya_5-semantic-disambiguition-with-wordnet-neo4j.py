# (On progress)



'''

This is the fifth homework of NLP course.



The aim of the homework is building a wordnet graph on neo4j from the point of view semantic relations 

between words. To use for disambiguation.



The wordnet is taken from nltk.corpus

'''



'''

Hocam bir konuşmamızdan sonra bu ödevi askıya alabileceğimi söylemiştiniz. Bundan dolayı şuan bitmiş

durumda değil. Son aşama olarak bu kodda çıkartmış olduğum veriler ile semantic relation kullanarak 

bir graph yaratmak kaldı. Bunu da yaptığımda sonlandırılmış halini size tekrar ileteceğim.

'''

import pandas as pd

from nltk.corpus import wordnet as wn

from py2neo import Graph, Node, Relationship
# For connection between neo4j

uri = "bolt://localhost:7687"

user = "neo4j"

password = "123"
graph = Graph(uri=uri, user=user, password=password)  # Connection. Named as "graph"



graph.delete_all()



cypher = graph.run



count = 0



df = pd.DataFrame(columns=["id", "syn"])



df2 = pd.DataFrame(columns=['id', 'synset'])



list1 = []

list2 = []

list3 = []
# Function get_word_synonyms_and_antonyms_from_wordnet: It finds the synonyms and antonyms of the word

# Parameter word: word from wordnet between 250 - 2500

# Parameter min_acceptable_reputation: desired reputation of synset

def get_word_synonyms_and_antonyms_from_wordnet(word, min_acceptable_reputation=1):

    synonyms = []

    antonyms = []



    word_synsets = wn.synsets(word)



    for syn in word_synsets:

        # find if there is a synset entry with desired min desired reputation

        synset_has_reputable_lemmas = False



        for lemma in syn.lemmas(): # Lemmas of synsets

            # check the reputation of this lemma, if not good enough, skip it

            if lemma.count() < min_acceptable_reputation:

                continue



            synset_has_reputable_lemmas = True # The lemma is known



            if lemma.name() != word:    # Be sure if the lemma is the different from the given word

                synonyms.append(lemma.name())



    if lemma.antonyms(): # For antonyms

        

        if lemma.antonyms()[0].name() != word: # if the antonym is the same as the word, pass it

            antonyms.append(lemma.antonyms()[0].name())

    if synset_has_reputable_lemmas:

        syn_name = syn.name()

        syn_name = syn_name.split(".")[0]

        if syn_name != word:

            synonyms.append(syn_name.replace("_", " "))



    synonyms = list(set(synonyms)) # List of synonyms

    antonyms = list(set(antonyms)) # List of antonyms

    return synonyms, antonyms
for word in wn.words():  # For all words in wordnet(vocabulary)

    if count == 2500:  # To make the execution faster, just look for between 250 - 2500

        break

    if count < 250:

        count += 1

        continue

    str = ''

    count += 1

    # if count < 240:

    #     continue

    syno = []

    anto = []

    syno, anto = get_word_synonyms_and_antonyms_from_wordnet(word)



    list1.append(count) # Put the count(as id) in list1

    list2.append(word) # Put the words for each time for a count



    for i in syno:  # Then for each synonym put count and synonym name

        list1.append(count)

        list2.append(i)
df2["id"] = list1

df2["synset"] = list2
list1 = list(dict.fromkeys(list1))
# This part is on progress, this part should contain the codes about building the graph

for i in list1:

    hold = df2.loc[df2['id'] == i]  # Picking for each same count(id)

    hold = hold.reset_index(drop=True)

    if len(hold) == 1:

        continue

    else:

        hold.iloc[0:]



    new_node = Node("dene", name=hold['synset'][0])  # Synonyms will be added

    graph.create(new_node)

    for y in hold['synset']:

        hold2 = df2.loc[df2['synset'] == y] 

        hold2 = hold2.reset_index(drop=True)

    for n in hold2['synset']:

        new_relation_node2 = Node("relation", name=n)

        graph.create(new_relation_node2)

        graph.create(Relationship(new_node, "Synset", new_relation_node2))