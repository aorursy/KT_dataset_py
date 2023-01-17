# Importing libraries
import numpy as np
import pandas as pd
import os
import pickle
import ast # parses list in lit string to pythong list
from tqdm import tqdm # progress bar helpful in monitoring processes
import decimal
# Def Load Files func
def loadfiles(directory):
    files = {} # Initiate file dict
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            if filename.split(".")[-1] == "csv": # load csv file
                files[''.join(filename.split(".")[:-1])] = pd.read_csv(fullpath)
                print(f"Loaded file: {filename}")
            elif filename.split(".")[-1] == "pkl": # load pkl file
                with open(fullpath, 'rb') as f:
                    files[''.join(filename.split(".")[:-1])] =  pickle.load(f)
                    print(f"Loaded file: {filename}")
    return files

# Load files
files = loadfiles('/kaggle/input')
ingredients = files['ingr_map']
recipes = files['RAW_recipes']
r2i_map_raw = files['PP_recipes']
# def method to clean r2i_map_raw table
def generate_maps(r2i_map_raw):
    r2i_map = {} # key = recipe id, value = ingredient id set
    i2r_map = {} # key = ingredient id, value = recipe id set

    # parse and append individual rows
    for i in tqdm(range(len(r2i_map_raw.id))):
        recipe_id = r2i_map_raw.id[i]
        
        # retrieve ingredients
        ingredients = ingredients = ast.literal_eval(r2i_map_raw.query(f"id == '{recipe_id}'").ingredient_ids.values[0])

        # add r2i entry
        r2i_map[recipe_id] = set(ingredients)

        # add i2r entry
        for i in ingredients:
            if i in i2r_map.keys():
                i2r_map[i] = i2r_map[i].union({recipe_id})
            else:
                i2r_map[i] = {recipe_id}
    
    return r2i_map, i2r_map

r2i_map, i2r_map = generate_maps(r2i_map_raw)

i2id_map_raw_replaced = ingredients[['id','replaced']].drop_duplicates(subset='replaced', keep="first")
i2id_map_raw_raw_ingr = ingredients[['id','raw_ingr']].drop_duplicates(subset='raw_ingr', keep="first")
i2id_map_raw_processed = ingredients[['id','processed']].drop_duplicates(subset='processed', keep="first")

i2id_map = {**dict(zip(list(i2id_map_raw_replaced['replaced']), list(i2id_map_raw_replaced['id']))),
            **dict(zip(list(i2id_map_raw_raw_ingr['raw_ingr']), list(i2id_map_raw_raw_ingr['id']))),
            **dict(zip(list(i2id_map_raw_processed['processed']), list(i2id_map_raw_processed['id'])))}

id2r_map_raw = recipes[['name','id']].drop_duplicates(subset='id', keep="first")
id2r_map = dict(zip(list(id2r_map_raw['id']),list(id2r_map_raw['name'])))

r2min_map_raw = recipes[['minutes','id']].drop_duplicates(subset='id', keep="first")
r2min_map = dict(zip(list(r2min_map_raw['id']),list(r2min_map_raw['minutes'])))

with open('/kaggle/working/i2r_map.pkl', 'wb') as handle:
    pickle.dump(i2r_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/kaggle/working/r2i_map.pkl', 'wb') as handle:
    pickle.dump(r2i_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/kaggle/working/i2id_map.pkl', 'wb') as handle:
    pickle.dump(i2id_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/kaggle/working/id2r_map.pkl', 'wb') as handle:
    pickle.dump(id2r_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/kaggle/working/r2min_map.pkl', 'wb') as handle:
    pickle.dump(r2min_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Loading previously generated mappers
with open('/kaggle/working/i2r_map.pkl', 'rb') as f:
    i2r_map =  pickle.load(f)

with open('/kaggle/working/r2i_map.pkl', 'rb') as f:
    r2i_map =  pickle.load(f)

with open('/kaggle/working/i2id_map.pkl', 'rb') as f:
    i2id_map =  pickle.load(f)

with open('/kaggle/working/id2r_map.pkl', 'rb') as f:
    id2r_map =  pickle.load(f)
    
with open('/kaggle/working/r2min_map.pkl', 'rb') as f:
    r2min_map =  pickle.load(f)
def getRecipes(ingredient_list_id):
    output_data = {} # key = recipe id, value = {'i_req': set(),'i_avail': set(),'i_needed': set(), 'time_req':r2min_map[r]}
    
    for i in ingredient_list_id:
        recipes = i2r_map[i] # Retrieve recipes containing this ingredient
        for r in recipes:
            if r in output_data.keys():
                output_data[r]['i_avail'] = output_data[r]['i_avail'].union({i})
            else:
                output_data[r] = {'i_req': r2i_map[r],'i_avail': {i}, 'time_req':r2min_map[r]}
    
    for r in output_data.keys():
        output_data[r]['i_needed'] = output_data[r]['i_req'].difference(output_data[r]['i_avail'])
    
    return output_data

def parseIngredientList(ingredient_list_string):
    ingredient_list_id=[]
    for i in ingredient_list_string:
        ingredient_list_id.append(i2id_map[i])
    return ingredient_list_id

def score(recipe_data):
    try:
        if recipe_data['time_req']==0: 
            return -decimal.Decimal(2)**decimal.Decimal(1000)
        else:
            score = (decimal.Decimal((len(recipe_data['i_avail']))**decimal.Decimal(60.0/float(recipe_data['time_req']))) - (decimal.Decimal(len(recipe_data['i_needed']))**decimal.Decimal(float(recipe_data['time_req'])/15)))
        return score
    except:
        return -decimal.Decimal(2)**decimal.Decimal(1000)

def sortByScore(output_data):
    return sorted(list(output_data.keys()), key=lambda recipe: score(output_data[recipe]), reverse=True)

def maxScoreRecipeId(output_data):
    return sortByScore(output_data)[0]

def getRecipeData(r_id,output_data):
    recipe_data_list = []
    recipe_data_list.append(r_id) # Append recipeId to list
    recipe_data_list.append(id2r_map[r_id]) # Append recipeName to list
    recipe_data_list.append(output_data[r_id]['time_req']) # Append prepTimeInMinutes to list
    recipe_data_list.append(len(output_data[r_id]['i_avail'])) # Append numberOfFridgeItemUsed to list
    recipe_data_list.append(len(output_data[r_id]['i_needed'])) # Append numberOfAdditionalItemsNeeded to list
    return recipe_data_list

def hungryWithMyFridgeAPI(arrayOfArrayOfIngredients):
    output_array = []
    for ingredientsArray in arrayOfArrayOfIngredients:
        recipes = getRecipes(parseIngredientList(ingredientsArray))
        output_array.append(getRecipeData(maxScoreRecipeId(recipes),recipes))
    return output_array

def scoreOutputArray(output_array):
    scores = []
    for output in output_array:
        scores.append((decimal.Decimal(output[3])**decimal.Decimal(60.0/output[2])) - (decimal.Decimal(output[4])**decimal.Decimal(output[2]/15)))
    return np.mean(scores)
## Evaluation

input = [
    ['winter squash', 'mexican seasoning', 'mixed spice', 'honey', 'butter', 'olive oil', 'salt'],
    ['low sodium chicken broth', 'tomatoes', 'zucchini', 'potatoes', 'wax beans', 'green beans', 'carrots'],
    ['spinach',  'garlic powder', 'soft breadcrumbs', 'oregano', 'onion'] ]
output = hungryWithMyFridgeAPI(input)

for i in range(len(input)):
    print("For Available ingredients:")
    print(input[i])
    print("\nRecommended Recipe:")
    print(output[i])
    print(f"With Score: {scoreOutputArray([output[i]])}\n")

print(f"\nOverall Mean Score: {scoreOutputArray(output)}")