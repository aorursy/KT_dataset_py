# Importing librarires

import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Reading data

df = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
df.head(20)
## Viewing unique ingredients to make sure that the data is clean

# Creating list from 'ingredients' column
ingredients_list = []
ingredients_list = df['ingredients'].tolist()

# Getting set from list
ingredients = []
for list in ingredients_list:
    temp = list.split(', ')
    for ingredient in temp:
        ingredients.append(ingredient.lower().strip())

unique_ingredients = sorted(set(ingredients))
# print(unique_ingredients)

## Cleaning ingredients column

# Lowercase
df['ingredients'] = df['ingredients'].str.lower()

# Removing white spaces
df['ingredients'] = df['ingredients'].str.strip()

def comma_space(x):
    x = x.replace(', ',',')
    return x

df['ingredients'] = df['ingredients'].apply(comma_space)
df.head(10)
# Creating key-value pairs (name: ingredients)
recipe_dict = dict(zip(df.name, df.ingredients))

clean_recipe_dict = {}
for recipe, ingredients in recipe_dict.items():
    temp = ingredients.split(",")
    clean_recipe_dict[recipe] = temp

# print(clean_recipe_dict)
# Creating recipe combinations

from itertools import combinations

recipe_combinations = []
for a,b in combinations(clean_recipe_dict.keys(), 2):
    recipe_combinations.append((a,b))
    
# Determining similarity score per recipe pair
# Score is determined as follows:
# num of ingredients in both a and b / num of ingredients in a

def similarity_score(a,b):
    ingredients_a = clean_recipe_dict[a]
    ingredients_b = clean_recipe_dict[b]
    num_similar = len(set(ingredients_a) & set(ingredients_b))
    num_a = len(ingredients_a)
    return num_similar/num_a

score_dict = {}
for combination in recipe_combinations:
    score = similarity_score(combination[0], combination[1])
    score_dict[combination] = score

# print(score_dict)
# EXAMPLE:
user_recipe = 'Kaju katli'
results = {}
for key in score_dict:
    if key[0] == user_recipe:
        results[key] = score_dict[key]

top_10_recipes = sorted(results, key=results.get, reverse=True)[:10]

results_with_scores = {}
for recipe in top_10_recipes:
    results_with_scores[recipe] = score_dict[recipe]

print(results_with_scores)
### UI: Choose a recipe to find the top 10 most similar recipes ###
# print("Enter a recipe name (case sensitive):")
# user_recipe = input()

# results = {}
# for key in score_dict:
#     if key[0] == user_recipe:
#         results[key] = score_dict[key]

# top_10_recipes = sorted(results, key=results.get, reverse=True)[:10]

# results_with_scores = {}
# for recipe in top_10_recipes:
#     results_with_scores[recipe] = score_dict[recipe]

# print(results_with_scores)