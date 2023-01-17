# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import sys



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



datafiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        ifile = os.path.join(dirname, filename)

        if ifile.split(".")[-1] == "json":

            datafiles.append(ifile)

        #print(ifile)



# Any results you write to the current directory are saved as output.
len(datafiles)
datafiles[0:5]
with open(datafiles[0], 'r')as f1:

    sample = json.load(f1)
for key,value in sample.items():

    print(key)
print(sample['metadata'].keys())

print('abstract: ',sample['abstract'][0].keys())

print('body_text: ',sample['body_text'][0].keys())

print('bib_entries: ',sample['bib_entries'].keys())

print('ref_entries: ', sample['ref_entries'].keys())

print('back_matter: ',sample['back_matter'][0].keys())
id2title = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    title = doc['metadata']['title']

    id2title.append({id:title})
#with open('id2title.json','w')as f2:

#    json.dump(id2title,f2)
id2abstract = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    abstract = ''

    for item in doc['abstract']:

        abstract = abstract + item['text']

        

    id2abstract.append({id:abstract})
id2abstract[0]
#with open('id2abstract.json','w')as f3:

#   json.dump(id2abstract,f3)
id2bodytext = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    bodytext = ''

    for item in doc['body_text']:

        bodytext = bodytext + item['text']

        

    id2bodytext.append({id:bodytext})
id2bodytext[0]
from termcolor import colored





def get_clor_entitis(sent, animals, coronas):

    colored_sent = sent

    

    for curr_animal in animals:

        colored_sent = colored_sent.replace(curr_animal, colored(curr_animal, 'blue', attrs=['bold']))

    

    for curr_corona in coronas:

        colored_sent = colored_sent.replace(curr_corona, colored(curr_corona, 'red', attrs=['bold']))





    return colored_sent






all_animal_set = set()

animal_list = [ "Aardvark", "Albatross", "Alligator", "Alpaca", "Ant", "Anteater", "Antelope", "Ape", "Armadillo", "Donkey",

               "Baboon", "Badger", "Barracuda", "Bat", "Bear", "Beaver", "Bee", "Bison", "Boar", "Buffalo", "Butterfly",

               "Camel", "Capybara", "Caribou", "Cassowary", "Cat", "Caterpillar", "Cattle", "Chamois", "Cheetah", "Chicken",

               "Chimpanzee", "Chinchilla", "Chough", "Clam", "Cobra", "Cockroach", "Cod", "Cormorant", "Coyote", "Crab",

               "Crane", "Crocodile", "Crow", "Curlew", "Deer", "Dinosaur", "Dog", "Dogfish", "Dolphin", "Dotterel", "Dove",

               "Dragonfly", "Duck", "Dugong", "Dunlin", "Eagle", "Echidna", "Eel", "Eland", "Elephant", "Elk", "Emu", "Falcon",

               "Ferret", "Finch", "Fish", "Flamingo", "Fly", "Fox", "Frog", "Gaur", "Gazelle", "Gerbil", "Giraffe", "Gnat", "Gnu",

               "Goat", "Goldfinch", "Goldfish", "Goose", "Gorilla", "Goshawk", "Grasshopper", "Grouse", "Guanaco", "Gull", "Hamster",

               "Hare", "Hawk", "Hedgehog", "Heron", "Herring", "Hippopotamus", "Hornet", "Horse", "Hummingbird", "Hyena", "Ibex",

               "Ibis", "Jackal", "Jaguar", "Jay", "Jellyfish", "Kangaroo", "Kingfisher", "Koala", "Kookabura", "Kouprey", "Kudu",

               "Lapwing", "Lark", "Lemur", "Leopard", "Lion", "Llama", "Lobster", "Locust", "Loris", "Louse", "Lyrebird", "Magpie",

               "Mallard", "Manatee", "Mandrill", "Mantis", "Marten", "Meerkat", "Mink", "Mole", "Mongoose", "Monkey", "Moose", 

               "Mosquito", "Mosquitoes", "Mouse", "Mule", "Narwhal", "Newt", "Nightingale", "Octopus", "Okapi", "Opossum", "Oryx", "Ostrich",

               "Otter", "Owl", "Oyster", "Panther", "Parrot", "Partridge", "Peafowl", "Pelican", "Penguin", "Pheasant", "Pig", 

               "Pigeon", "Pony", "Porcupine", "Porpoise", "Quail", "Quelea", "Quetzal", "Rabbit", "Raccoon", "Rail", "Ram", "Rat",

               "Raven", "Red deer", "Red panda", "Reindeer", "Rhinoceros", "Rook", "Salamander", "Salmon", "Sand Dollar", "Sandpiper",

               "Sardine", "Scorpion", "Seahorse", "Seal", "Shark", "Sheep", "Shrew", "Skunk", "Snail", "Snake", "Sparrow", "Spider",

               "Spoonbill", "Squid", "Squirrel", "Starling", "Stingray", "Stinkbug", "Stork", "Swallow", "Swan", "Tapir", "Tarsier", 

               "Termite", "Tiger", "Toad", "Trout", "Turkey", "Turtle", "Viper", "Vulture", "Wallaby", "Walrus", "Wasp", "Weasel", 

               "Whale", "Wildcat", "Wolf", "Wolverine", "Wombat", "Woodcock", "Woodpecker", "Worm", "Wren", "Yak", "Zebra" ]



corona_names = {'corona', 'covid19', 'coronavirus', 'corona', 'coronaviruses', 'CoV'}



for animal in animal_list:

    all_animal_set.add(animal.lower())

    all_animal_set.add(animal.lower() + 's')

    

is_animale = False



count_animals = 0

count_animals_per_doc = 0

cc = 0

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    bodytext = ''

    if is_animale:

        count_animals_per_doc += 1

    is_animale = False

    for item in doc['body_text']:

        

        animals_in_item = set([tok for tok in item['text'].split(" ") if tok.lower() in all_animal_set])

        corona_names_in_item = set([tok for tok in item['text'].split(" ") if tok.lower() in corona_names])

        

        if animals_in_item and corona_names_in_item: 

            count_animals += 1

            is_animale = True

#             print(item['text'])

            print(get_clor_entitis(item['text'], animals_in_item, corona_names_in_item))

            print(animals_in_item, corona_names_in_item)

            print("--------------------"+str(count_animals)+"-------------------------")

            print()

            cc += 1

            if cc == 200:

                sys.exit()

            break

       

        

        bodytext = bodytext + item['text']

        

print(count_animals)

print(count_animals_per_doc)    

    

    






all_animal_set = set()

animal_list = [ "Aardvark", "Albatross", "Alligator", "Alpaca", "Ant", "Anteater", "Antelope", "Ape", "Armadillo", "Donkey",

               "Baboon", "Badger", "Barracuda", "Bat", "Bear", "Beaver", "Bee", "Bison", "Boar", "Buffalo", "Butterfly",

               "Camel", "Capybara", "Caribou", "Cassowary", "Cat", "Caterpillar", "Cattle", "Chamois", "Cheetah", "Chicken",

               "Chimpanzee", "Chinchilla", "Chough", "Clam", "Cobra", "Cockroach", "Cod", "Cormorant", "Coyote", "Crab",

               "Crane", "Crocodile", "Crow", "Curlew", "Deer", "Dinosaur", "Dog", "Dogfish", "Dolphin", "Dotterel", "Dove",

               "Dragonfly", "Duck", "Dugong", "Dunlin", "Eagle", "Echidna", "Eel", "Eland", "Elephant", "Elk", "Emu", "Falcon",

               "Ferret", "Finch", "Fish", "Flamingo", "Fly", "Fox", "Frog", "Gaur", "Gazelle", "Gerbil", "Giraffe", "Gnat", "Gnu",

               "Goat", "Goldfinch", "Goldfish", "Goose", "Gorilla", "Goshawk", "Grasshopper", "Grouse", "Guanaco", "Gull", "Hamster",

               "Hare", "Hawk", "Hedgehog", "Heron", "Herring", "Hippopotamus", "Hornet", "Horse", "Hummingbird", "Hyena", "Ibex",

               "Ibis", "Jackal", "Jaguar", "Jay", "Jellyfish", "Kangaroo", "Kingfisher", "Koala", "Kookabura", "Kouprey", "Kudu",

               "Lapwing", "Lark", "Lemur", "Leopard", "Lion", "Llama", "Lobster", "Locust", "Loris", "Louse", "Lyrebird", "Magpie",

               "Mallard", "Manatee", "Mandrill", "Mantis", "Marten", "Meerkat", "Mink", "Mole", "Mongoose", "Monkey", "Moose", 

               "Mosquito", "Mosquitoes", "Mouse", "Mule", "Narwhal", "Newt", "Nightingale", "Octopus", "Okapi", "Opossum", "Oryx", "Ostrich",

               "Otter", "Owl", "Oyster", "Panther", "Parrot", "Partridge", "Peafowl", "Pelican", "Penguin", "Pheasant", "Pig", 

               "Pigeon", "Pony", "Porcupine", "Porpoise", "Quail", "Quelea", "Quetzal", "Rabbit", "Raccoon", "Rail", "Ram", "Rat",

               "Raven", "Red deer", "Red panda", "Reindeer", "Rhinoceros", "Rook", "Salamander", "Salmon", "Sand Dollar", "Sandpiper",

               "Sardine", "Scorpion", "Seahorse", "Seal", "Shark", "Sheep", "Shrew", "Skunk", "Snail", "Snake", "Sparrow", "Spider",

               "Spoonbill", "Squid", "Squirrel", "Starling", "Stingray", "Stinkbug", "Stork", "Swallow", "Swan", "Tapir", "Tarsier", 

               "Termite", "Tiger", "Toad", "Trout", "Turkey", "Turtle", "Viper", "Vulture", "Wallaby", "Walrus", "Wasp", "Weasel", 

               "Whale", "Wildcat", "Wolf", "Wolverine", "Wombat", "Woodcock", "Woodpecker", "Worm", "Wren", "Yak", "Zebra" ]



corona_names = {'corona', 'covid19', 'coronavirus', 'corona'}



for animal in animal_list:

    all_animal_set.add(animal.lower())

    all_animal_set.add(animal.lower() + 's')

    

is_animale = False



count_animals = 0

count_no = 0



print(len(datafiles))



for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)



    animals_in_item = set([tok for item in doc['body_text'] for tok in item['text'].split(" ") if tok.lower() in all_animal_set])

    corona_names_in_item = set([tok for item in doc['body_text'] for tok in item['text'].split(" ") if tok.lower() in corona_names])



    if animals_in_item and corona_names_in_item: 

        count_animals += 1

    else:

        count_no += 1



        

print(count_animals)

print(count_no)

    
with open('id2bodytext.json','w')as f4:

   json.dump(id2bodytext,f4)
bibEntries = []

for key,value in sample['bib_entries'].items():

    refid = key

    title = value['title']

    year = value['year']

    venue = value['venue']

    try:

        DOI = value['other_ids']['DOI'][0]

    except:

        DOI = 'NA'

        

    bibEntries.append({"refid": refid,\

                      "title":title,\

                      "year": year,\

                      "venue":venue,\

                      "DOI": DOI})
bibEntries[0:5]
import networkx as nx
G = nx.Graph()

G.add_node(sample['paper_id'])

for item in bibEntries:

    G.add_node(item['DOI'], title = item['title'], year = item['year'], venue = item['venue'])

    G.add_edge(sample['paper_id'], item['DOI'], value = item['refid'])  
len(G.nodes())
import matplotlib.pyplot as plt
for item in list(G.nodes().data('venue')):

    print(item)
#for item in list(G.nodes().data('title')):

#    print(item)
#for item in list(G.nodes().data('year')):

#    print(item)