!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@nlp
import sys

sys.path.append('/kaggle/working')
import pandas as pd



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex1 import *

print("\nSetup complete")
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
import spacy

from spacy.matcher import PhraseMatcher



# Load the SpaCy model

nlp = spacy.load('en_core_web_sm')

# Create the doc object

review_doc = nlp(data.iloc[4].text)



# Create the PhraseMatcher object, be sure to match on lowercase text

matcher = ____



# Create a list of docs for each item in the menu

patterns = ____



# Add the item patterns to the matcher

____



# Find matches in the review_doc

matches = ____
# Uncomment if you need some guidance

# q_1.hint()

# q_1.solution()
# After implementing the above cell, uncomment and run this to print 

# out the matches. Otherwise you'll get an error.



# for match in matches:

#     print(f"At position {match[1]}: {review_doc[match[1]:match[2]]}")
#%%RM_IF(PROD)%%



import spacy

from spacy.matcher import PhraseMatcher



nlp = spacy.load('en_core_web_sm')

review_doc = nlp(data.iloc[4].text)



matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

patterns = [nlp(item) for item in menu]

matcher.add("MENU", None, *patterns)

matches = matcher(review_doc)



for match in matches:

    print(f"At position {match[1]}: {review_doc[match[1]:match[2]]}")

    

# Uncomment when checking code is complete

q_1.assert_check_passed()
from collections import defaultdict



# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,

# the key is added with an empty list as the value.

item_ratings = defaultdict(list)



for idx, review in data.iterrows():

    doc = ____

    # Using the matcher from the previous exercise

    matches = ____

    

    # Create a set of the items found in the review text

    found_items = ____

    

    # Update item_ratings with rating for each item in found_items

    # Transform the item strings to lowercase to make it case insensitive

    ____



q_2.check()
# Uncomment if you need some guidance

#q_2.hint()

#q_2.solution()
#%%RM_IF(PROD)%%



from collections import defaultdict



item_ratings = defaultdict(list)



for idx, review in data.iterrows():

    doc = nlp(review.text)

    matches = matcher(doc)



    found_items = set([doc[match[1]:match[2]] for match in matches])

    

    for item in found_items:

        item_ratings[str(item).lower()].append(review.stars)

        

q_2.assert_check_passed()
similar_items = [('cheesesteak', 'cheese steak'),

                 ('cheesesteak', 'steak and cheese'),

                 ('chicken parmigiana', 'chicken parm'),

                 ('chicken parmigiana', 'chicken parmesan'),

                 ('mac and cheese', 'macaroni'),

                 ('calzone', 'calzones')]



for (destination, source) in similar_items:

    item_ratings[destination].extend(item_ratings.pop(source))
# Calculate the mean ratings for each menu item as a dictionary

mean_ratings = ____



# Sort the ratings in descending order, should be a list

best_items = ____



q_3.check()
# Uncomment if you need some guidance

# q_3.hint()

# q_3.solution()
# After implementing the above cell, uncomment and run this to print 

# out the best items. Otherwise you'll get an error.



# for item in best_items:

#     print(f"{item:>25}{mean_ratings[item]:>10.3f}")
#%%RM_IF(PROD)%%



mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}

best_items = sorted(mean_ratings, key=mean_ratings.get, reverse=True)



for item in best_items:

    print(f"{item:>25}{mean_ratings[item]:>10.3f}")

    

q_3.assert_check_passed()
counts = {item: len(ratings) for item, ratings in item_ratings.items()}
item_counts = sorted(counts, key=counts.get, reverse=True)

for item in item_counts:

    print(f"{item:>25}{counts[item]:>5}")
#q_4.solution()
print("Best rated menu items:")

for item in best_items[:10]:

    print(f"{item:20} Average rating: {mean_ratings[item]:.3f} \tcount: {counts[item]}")
print("Worst rated menu items:")

for item in best_items[:-10:-1]:

    print(f"{item:20} Average rating: {mean_ratings[item]:.3f} \tcount: {counts[item]}")