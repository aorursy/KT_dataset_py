# 0. Initialization
# 0.1 Imports
print("\n# 0.1 Imports\n")
import numpy as np # NumPy for linear algebra
import pandas as pd # Pandas for data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb # Seaborn for statistic viz
import os # OS for file management
from collections import Counter # Counter to build dictionnries given the count of elements
from itertools import chain, combinations # chain to build counters from lists, combinations for faster than two nested loops code

# 0.2 Read training data in a Pandas DataFrame
print("\n# 0.2 Read training data in a Pandas DataFrame\n")
print(os.listdir("../input")) # List input directory content
training_file_name = "train.json"
training_file = pd.read_json("../input/" + str(training_file_name))

# 0.3 Analyse DataFrame content
print("\n# 0.3 Analyse DataFrame content\n")
print(training_file.head())
print(training_file.dtypes)

# 0.4 Find and remove null data in DataFrame
print("\n# 0.4 Find and remove null data in DataFrame\n")
training_file.isnull().sum()
# 1. Analysis of data content
# 1.1 Find the different types of cusine stored in the data
print("\n# 1.1 Find the different types of cuisine stored in the data\n")
single_cuisines = training_file.cuisine.unique()
print(single_cuisines)
n_single_cuisines = len(single_cuisines)
print("Number of cuisines = "+str(n_single_cuisines))

# 1.2 Find the different ingredients stored in the data
print("\n# 1.2 Find the different ingredients stored in the data\n")
counted_ingredients = Counter(chain.from_iterable(training_file.ingredients.tolist())) # Builds a counter to count the occurences of each single ingredient
single_ingredients = list(counted_ingredients.keys()) # Builds a list will all single ingredients
n_single_ingredients = len(single_ingredients) # Counts the number of single ingredients
print("Number of single ingredients = " + str(n_single_ingredients))

# 1.3 Find the ingredients that are typical of one cuisine and find ingredients by cuisine
print("\n# 1.3 Find the ingredients that are typical of one cuisine and find ingredients by cuisine\n")
specific_ingredients = {} # Will be a dict of lists
sorted_ingredients = {} # Will be a dict of Counters
sorted_not_specific_ingredients = {} # Will be a dict of lists
not_specific_ingredients = single_ingredients # Will be a list

for cuisine in single_cuisines :
    sorted_ingredients[cuisine] = Counter(chain.from_iterable(training_file[training_file.cuisine == cuisine].ingredients.tolist()))
    other_cuisines_ingredients = Counter(chain.from_iterable(training_file[training_file.cuisine != cuisine].ingredients.tolist()))
    intersection  = Counter(sorted_ingredients[cuisine]) & Counter(other_cuisines_ingredients)
    specific_ingredients[cuisine] = list(sorted_ingredients[cuisine] - intersection)
    not_specific_ingredients = list(Counter(not_specific_ingredients) - Counter(specific_ingredients[cuisine]))
    sorted_not_specific_ingredients[cuisine] = list(sorted_ingredients[cuisine] - Counter(specific_ingredients[cuisine]))
    # Print info about current cuisine
    print(cuisine + " cuisine has " + str(len(specific_ingredients[cuisine])) + " specific ingredients and uses "+ str(len(sorted_ingredients[cuisine])) + "  different ingredients")
 
sum_specific_ingredients = sum(len(v) for v in specific_ingredients.values())
sum_not_specific_ingredients = len(not_specific_ingredients)

print("\n" + str(sum_specific_ingredients) + " ingredients are specific to one cuisine and \n" + str(sum_not_specific_ingredients)+" ingredients are not specific to one cuisine")

# 1.4 Count recipe by cuisine
print("\n# 1.4 Count recipe by cuisine\n")
recipe_by_cuisine = training_file.groupby('cuisine').cuisine.count().to_dict()
# 2. Train a model using a scoring matrix
# 2.1 Build the matrix
"""
    Here we build an integer matrix within a DataFrame with all ingredients as lines (6714) columns are various cuisines (20)
"""
DF_score = pd.DataFrame({'ingredient_name': single_ingredients})
DF_matrix = pd.DataFrame(np.zeros((n_single_ingredients, n_single_cuisines), dtype = 'int'), columns=single_cuisines)
DF_score = pd.concat([DF_score,DF_matrix], axis=1)

# 2.2 Fill the matrix coefficients
for cuisine in single_cuisines :
    # See if something faster than those two loops can be done
    for ingredient in sorted_not_specific_ingredients[cuisine]:
        DF_score.loc[DF_score['ingredient_name'] == ingredient, cuisine] = sorted_ingredients[cuisine][ingredient]/recipe_by_cuisine[cuisine] # Pounded by the number of reciepe by cuisine
    for ingredient in specific_ingredients[cuisine]:
        DF_score.loc[DF_score['ingredient_name'] == ingredient, cuisine] = 100*sorted_ingredients[cuisine][ingredient]/recipe_by_cuisine[cuisine] # Ingredients that are specific to one cuisine increase te probability of the dish of being of this cuisine
print(DF_score.head())
DF_score.loc[DF_score['ingredient_name'] == 'olive oil', :]
# 3.1 Function to evaluate the score of a given ingredient list
def evaluate_cuisine_score(ing_list):
    score = dict.fromkeys(single_cuisines,0) # Builds a dict full of zeros (unknown performance)
    for ing in ing_list:
        for cuisine in single_cuisines:#performance is K*len(ing_list)*len(single_cuisines)
            if not DF_score.loc[DF_score['ingredient_name'] == ing, cuisine].empty: # Check that dataset is not empty (ie. check that the ingredient has been ranked)
                score[cuisine] += DF_score.loc[DF_score['ingredient_name'] == ing, cuisine].item()
    return score
# 3.2 Function which returns the cuisines having the maximal score (if more than one, they are returned as a list)
def find_cuisine_from_score(score): 
    return [key for m in [max(score.values())] for key,val in score.items() if val == m] # list containing the keys of the highest values of the score dict

# 3.3 Function which finds the possible cusisines of an ingredient list
def find_cuisine(ing_list):
    return find_cuisine_from_score(evaluate_cuisine_score(ing_list))
# 3.4 Tests
# 3.4.1 Ingredients list for testing
import matplotlib.pyplot as plt
ing_list = ['bread machine yeast','beni shoga','salt','olive oil','meat cuts']
find_cuisine(ing_list)
D = evaluate_cuisine_score(ing_list)
plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()), rotation='vertical')
print (str(find_cuisine(ing_list)))
# 3.4.2 Execution time of one test
from time import time
start = time()
find_cuisine(ing_list)
end = time()
print(end - start)
# 100. Test Kernel
# 100.1 read DataFrame content
print("\n# 100.1 read DataFrame content\n")
test_file_name = "test.json"
test_file = pd.read_json("../input/" + str(test_file_name))
print(len(test_file))
#test_file = test_file.head(10)

# 100.2 Predict cuisine
start = time()
test_file["cuisine_guesses"] = None
for row in test_file.itertuples() :
    test_file.at[row.Index, "cuisine_guesses"] = find_cuisine(test_file.loc[row.Index, "ingredients"])
end = time()
print("Done - elapsed time = "+str(end - start))
print(test_file)
