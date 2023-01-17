import pandas as pd



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex1 import *

print('Setup Complete')
# Load in the data from JSON file

data = pd.read_json('../input/nlp-course/restaurant.json')

data.head()
menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",

        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",

        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",

        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",

        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",

        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",

        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",

        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  

         "Prosciutto", "Salami"]
# Check your answer (Run this code cell to receive credit!)

q_1.solution()
import spacy

from spacy.matcher import PhraseMatcher



index_of_review_to_test_on = 14

text_to_test_on = data.text.iloc[index_of_review_to_test_on]



# Load the SpaCy model

nlp = spacy.blank('en')



# Create the tokenized version of text_to_test_on

review_doc = nlp(text_to_test_on)



# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')



# Create a list of tokens for each item in the menu

menu_tokens_list = [nlp(item) for item in menu]



# Add the item patterns to the matcher. 

# Look at https://spacy.io/api/phrasematcher#add in the docs for help with this step

# Then uncomment the lines below 



# 

matcher.add("MENU",            # Just a name for the set of rules we're matching to

            None,              # Special actions to take on matched words

            *menu_tokens_list)



#Find matches in the review_doc

matches = matcher(review_doc)



# Uncomment to check your work

#q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

q_2.solution()
for match in matches:

    print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")
from collections import defaultdict



# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,

# the key is added with an empty list as the value.

item_ratings = defaultdict(list)



for idx, review in data.iterrows():

    doc = nlp(review['text'])

    # Using the matcher from the previous exercise

    matches = matcher(doc)

    #print(doc[137:138], matches)

    # Create a set of the items found in the review text

    found_items = [doc[match[1]:match[2]] for match in matches]

    

    # Update item_ratings with rating for each item in found_items

    # Transform the item strings to lowercase to make it case insensitive

    val = review['stars']

    for x in found_items:

        x = str(x).lower()

        item_ratings[x].append(val)



q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Calculate the mean ratings for each menu item as a dictionary

mean_ratings = [[sum(item_ratings[x])/len(item_ratings[x]),x] for x in item_ratings.keys()]



# Find the worst item, and write it as a string in worst_text. This can be multiple lines of code if you want.

worst_item = min(mean_ratings)[1]

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()

len(mean_ratings)
# After implementing the above cell, uncomment and run this to print 

# out the worst item, along with its average rating. 



print(worst_item)

print([mean_ratings[x][0] for x in range(len(mean_ratings)) if mean_ratings[x][1]==worst_item])
counts = {item: len(ratings) for item, ratings in item_ratings.items()}



item_counts = sorted(counts, key=counts.get, reverse=True)

for item in item_counts:

    print(f"{item:>25}{counts[item]:>5}")
pass

sorted_ratings = sorted(mean_ratings)



print("Worst rated menu items:")

for item in sorted_ratings[:10]:

    print(f"{item[1]:20} Ave rating: {item[0]:.2f} \tcount: {counts[item[1]]}")

    

print("\n\nBest rated menu items:")

for item in sorted_ratings[-10:]:

    print(f"{item[1]:20} Ave rating: {item[0]:.2f} \tcount: {counts[item[1]]}")
# Check your answer (Run this code cell to receive credit!)

q_5.solution()