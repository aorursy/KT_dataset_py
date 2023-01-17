# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 as sql

from textblob import TextBlob as tb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
conn = sql.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')

db = pd.read_sql(

                       """

                       

                        SELECT *

                        from Reviews 

                        

                       """, con=conn)



print(db)
reviews = pd.read_sql(

                       """

                       

                        SELECT Id, ProductID, Text

                        from Reviews

                        

                       """, con=conn)



print(reviews)
opinion = tb("This is the best food I've ever had")

print(opinion)

print(opinion.sentiment)

print()



halfway_opinion = tb("This is good food")

print(halfway_opinion)

print(halfway_opinion.sentiment)

print()



statement = tb("This is food")

print(statement)

print(statement.sentiment)

print()



multiple = tb("I liked this product. It tasted great. However, I wish it didn't have a strong bitter taste.")

print(multiple)

print(multiple.sentiment)

print()

for sentence in multiple.sentences:

    print(sentence)

    print(sentence.sentiment)

    print()
from textblob import Word



word = Word("food")



print(word.synsets)

print()

print(word.definitions)

print()



food = word.synsets[0] # saves our intention for food as the first definition



print(food.lemma_names()) #prints synonyms of synset
print(food.hypernyms()) #more general

print()

print (food.hyponyms()) #more specific
print(food.member_holonyms()) #food is contained in these items

print()

print(food.part_meronyms()) #food is made up of these items

print()
from textblob.wordnet import Synset



macaroni = Word("macaroni")

print(macaroni.synsets)

print(macaroni.definitions)

macaroni = macaroni.synsets[1]

print(macaroni)

print()



print(macaroni.path_similarity(macaroni)) #should be identical

print()



print(macaroni.path_similarity(food)) #should be at least somewhat similar



pasta = Word("pasta")

print(pasta.synsets)

print(pasta.definitions)

max = -1

for version in pasta.synsets: #for each definition of pasta, pick the one that is most similar to macaroni and save that as "pasta"

    if macaroni.path_similarity(version) > max:

        pasta = version

        max = macaroni.path_similarity(version)



print(pasta)

print(macaroni.path_similarity(pasta))

print()



cheese = Word("cheese")

print(cheese.synsets)

print(cheese.definitions)

max = -1

for version in cheese.synsets:

    if macaroni.path_similarity(version) == None:

        continue

    if macaroni.path_similarity(version) > max:

        cheese = version

        max = macaroni.path_similarity(version)

    



print(cheese)



print(macaroni.path_similarity(cheese))