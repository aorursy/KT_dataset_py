# setup code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex1 import *

print('Setup is completed.')
# load in the data from JSON file

import pandas as pd

data = pd.read_json('../input/nlp-course/restaurant.json')

data.head()
menu = [

    "Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",

    "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",

    "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",

    "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",

    "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",

    "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",

    "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",

    "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  

    "Prosciutto", "Salami"

]
# check your answer (run this code cell to receive credit!)

q_1.solution()
index_of_review_to_test_on = 14

text_to_test_on = data['text'].iloc[index_of_review_to_test_on]



# create a blank model of a given language class. this function is the twin of `spacy.load()`

import spacy

nlp = spacy.blank('en')



# create the tokenized version of text_to_test_on

review_doc = nlp(text_to_test_on)



# create the PhraseMatcher object, to match a list of terms

from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(

    nlp.vocab,          # tokenizer

    attr='LOWER'        # to make consistent capitalization

)



# create a list of tokens for each item in the menu

menu_tokens_list = [nlp(item) for item in menu]



# add the item patterns to the matcher, for more help: https://spacy.io/api/phrasematcher#add

matcher.add(

    "MENU",            # Just a name for the set of rules we're matching to

    None,              # Special actions to take on matched words

    *menu_tokens_list  

)



# find matches in the review_doc

matches = matcher(review_doc)



# check your work

q_2.check()
# lines below will give you a hint or solution code

# q_2.hint()

# q_2.solution()
print(review_doc)

print('-'*10)

print(matches)

print('-'*10)

for match in matches:

   print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")
from collections import defaultdict



# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,

# the key is added with an empty list as the value.

item_ratings = defaultdict(list)



for idx, review in data.iterrows():

    doc = nlp(review['text'])

    

    # using the matcher from the previous exercise

    matches = matcher(doc)

    

    # create a set of the items found in the review text

    found_items = set([doc[match[1]:match[2]] for match in matches])

    

    # Update item_ratings with rating for each item in found_items

    # Transform the item strings to lowercase to make it case insensitive

    for item in found_items:

        item_ratings[str(item).lower()].append(review['stars'])



# check your work

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
# calculate the mean ratings for each menu item as a dictionary

from statistics import mean

mean_ratings = {item: mean(ratings) for item, ratings in item_ratings.items()}



# find the worst item, and write it as a string in worst_text

worst_item = min(mean_ratings, key=mean_ratings.get)



# check your work

q_4.check()
# lines below will give you a hint or solution code

# q_4.hint()

# q_4.solution()
# print out the worst item, along with its average rating. 

print(worst_item)

print(mean_ratings[worst_item])
counts = {item: len(ratings) for item, ratings in item_ratings.items()}



item_counts = sorted(counts, key=counts.get, reverse=True)

for item in item_counts:

    print(f"{item:>25}{counts[item]:>5}")
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)



print("Worst rated menu items:")

for item in sorted_ratings[:10]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")

    

print("\n\nBest rated menu items:")

for item in sorted_ratings[-10:]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
# check your answer (run this code cell to receive credit!)

q_5.solution()
# same output + errors

sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)



print("Worst rated menu items:")

for item in sorted_ratings[:10]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} ± {1/(counts[item] ** 0.5):.2f} \tcount: {counts[item]}")

    

print("\n\nBest rated menu items:")

for item in sorted_ratings[-10:]:

    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} ± {1/(counts[item] ** 0.5):.2f} \tcount: {counts[item]}")