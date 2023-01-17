# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import tensorflow as tf

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

searfor=['achiote', 'acid', 'agar', 'agave', 'alcohol', 'alfredo sauce', 'amaranth', 'amarena cherries', 'amaretti cookies', 'amber', 'anaheim chile', 'anardana', 'anchovies', 'anise', 'apple', 'apricot', 'arhar', 'artichoke', 'asafoetida', 'asian herb', 'asparagus', 'avocado', 'bacon', 'bagel', 'baguette', 'baking powder', 'balsamico', 'bamboo', 'banana', 'banh', 'barbecue sauce', 'barley', 'basil', 'bass', 'bay leaves', 'bean', 'bechamel', 'beef', 'beer', 'beetroot', 'biscuits', 'bisquick', 'bitters', 'blood orange', 'blueberry', 'bone', 'bonito', 'bordelaise sauce', 'bouillon', 'bread', 'broccoli', 'broth', 'butter', 'buttermilk', 'cabbage', 'cactus', 'calamari', 'candied', 'candlenuts', 'candy', 'canola oil', 'cardamon', 'carrot', 'catfish', 'caviar', 'celery', 'cereal', 'cheese', 'chicken', 'chile', 'chili', 'chili sauce', 'chipotle', 'chips', 'chive', 'chocolate', 'chocolate milk', 'ciabatta', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'codfish', 'coffee', 'cola', 'condensed cream', 'cookie', 'coriander', 'corn', 'corn starch', 'crab legs', 'cranberry', 'crepes', 'cucumber', 'cumin', 'curry', 'dill', 'dough', 'dragon fruit', 'dressing', 'duck', 'dumplings', 'eel ', 'egg', 'eggplant', 'falafel', 'fat', 'feta', 'fig', 'filet', 'fish', 'flax', 'flour', 'fries', 'garlic', 'gelatin', 'ginger', 'gnocchi', 'goat', 'goose', 'grape', 'habanero', 'ham', 'hazelnuts', 'heavy cream', 'honey', 'horse', 'hot dog', 'hot sauce', 'ice cream', 'jalapeno', 'jasmin', 'jelly', 'jerk', 'ketchup', 'kiwi', 'kung pao sauce', 'lamb', 'lasagna', 'lasagne', 'lavender', 'lemon', 'lettuce', 'lime juice', 'liqueur', 'loaf', 'lobster', 'malt', 'mandarin', 'mango', 'masala', 'mayonnaise', 'meat', 'melon', 'milk', 'mint', 'miso', 'mushroom', 'mustard', 'noodle', 'nut', 'nutella', 'oil', 'olive oil', 'olives', 'onion', 'orange', 'oregano', 'oyster sauce', 'pancake', 'paprika', 'parsley', 'passata', 'pasta', 'peach', 'peas', 'pecan', 'pepper', 'pepperoni', 'pesto', 'pig', 'pineapple', 'pistachio', 'pizza', 'pork', 'potato', 'pudding', 'pumpkin', 'rabbit', 'radish', 'ragu', 'raisin', 'raspberry', 'ribs', 'rice', 'rice noodles', 'salami', 'salmon', 'salsa', 'salt', 'sardines', 'sauerkraut', 'sausage', 'scallop', 'schmaltz', 'seafood', 'seaweed', 'seed', 'sesame', 'shallot', 'sherry', 'shrimp', 'sour cream', 'soy', 'soy sauce', 'soymilk', 'spaghetti', 'spinach', 'sprout', 'steak', 'stew', 'stock', 'strawberries', 'sugar', 'sweetener', 'syrup', 'tamarind', 'tart', 'tea ', 'thyme', 'toast', 'tofu', 'tomato', 'tomato sauce', 'tortilla', 'truffle', 'tumeric', 'tuna', 'turkey', 'vanilla', 'vinegar', 'wasabi', 'water', 'wheat', 'whipping cream', 'whiskey', 'wine', 'worcester', 'yeast', 'yogurt', 'yolk', 'zest', 'zucchini']



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ferdinant = input()

for i in range(0,len(searfor)):

    if(ferdinant in searfor[i]):

        print(i+1)

        print(searfor[i])     

searfor=['achiote', 'acid', 'agar', 'agave', 'alcohol', 'alfredo sauce', 'amaranth', 'amarena cherries', 'amaretti cookies', 'amber', 'anaheim chile', 'anardana', 'anchovies', 'anise', 'apple', 'apricot', 'arhar', 'artichoke', 'asafoetida', 'asian herb', 'asparagus', 'avocado', 'bacon', 'bagel', 'baguette', 'baking powder', 'balsamico', 'bamboo', 'banana', 'banh', 'barbecue sauce', 'barley', 'basil', 'bass', 'bay leaves', 'bean', 'bechamel', 'beef', 'beer', 'beetroot', 'biscuits', 'bisquick', 'bitters', 'blood orange', 'blueberry', 'bone', 'bonito', 'bordelaise sauce', 'bouillon', 'bread', 'broccoli', 'broth', 'butter', 'buttermilk', 'cabbage', 'cactus', 'calamari', 'candied', 'candlenuts', 'candy', 'canola oil', 'cardamon', 'carrot', 'catfish', 'caviar', 'celery', 'cereal', 'cheese', 'chicken', 'chile', 'chili', 'chili sauce', 'chipotle', 'chips', 'chive', 'chocolate', 'chocolate milk', 'ciabatta', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'codfish', 'coffee', 'cola', 'condensed cream', 'cookie', 'coriander', 'corn', 'corn starch', 'crab legs', 'cranberry', 'crepes', 'cucumber', 'cumin', 'curry', 'dill', 'dough', 'dragon fruit', 'dressing', 'duck', 'dumplings', 'eel ', 'egg', 'eggplant', 'falafel', 'fat', 'feta', 'fig', 'filet', 'fish', 'flax', 'flour', 'fries', 'garlic', 'gelatin', 'ginger', 'gnocchi', 'goat', 'goose', 'grape', 'habanero', 'ham', 'hazelnuts', 'heavy cream', 'honey', 'horse', 'hot dog', 'hot sauce', 'ice cream', 'jalapeno', 'jasmin', 'jelly', 'jerk', 'ketchup', 'kiwi', 'kung pao sauce', 'lamb', 'lasagna', 'lasagne', 'lavender', 'lemon', 'lettuce', 'lime juice', 'liqueur', 'loaf', 'lobster', 'malt', 'mandarin', 'mango', 'masala', 'mayonnaise', 'meat', 'melon', 'milk', 'mint', 'miso', 'mushroom', 'mustard', 'noodle', 'nut', 'nutella', 'oil', 'olive oil', 'olives', 'onion', 'orange', 'oregano', 'oyster sauce', 'pancake', 'paprika', 'parsley', 'passata', 'pasta', 'peach', 'peas', 'pecan', 'pepper', 'pepperoni', 'pesto', 'pig', 'pineapple', 'pistachio', 'pizza', 'pork', 'potato', 'pudding', 'pumpkin', 'rabbit', 'radish', 'ragu', 'raisin', 'raspberry', 'ribs', 'rice', 'rice noodles', 'salami', 'salmon', 'salsa', 'salt', 'sardines', 'sauerkraut', 'sausage', 'scallop', 'schmaltz', 'seafood', 'seaweed', 'seed', 'sesame', 'shallot', 'sherry', 'shrimp', 'sour cream', 'soy', 'soy sauce', 'soymilk', 'spaghetti', 'spinach', 'sprout', 'steak', 'stew', 'stock', 'strawberries', 'sugar', 'sweetener', 'syrup', 'tamarind', 'tart', 'tea ', 'thyme', 'toast', 'tofu', 'tomato', 'tomato sauce', 'tortilla', 'truffle', 'tumeric', 'tuna', 'turkey', 'vanilla', 'vinegar', 'wasabi', 'water', 'wheat', 'whipping cream', 'whiskey', 'wine', 'worcester', 'yeast', 'yogurt', 'yolk', 'zest', 'zucchini']



grob=0

model_embedding=tf.keras.models.load_model('../input/v2-of-my-recipe-ai/ok.h5')





Z=[26]

bool_real=True

const_bool_real=0

def klein(terror=[]):

    searfor=['achiote', 'acid', 'agar', 'agave', 'alcohol', 'alfredo sauce', 'amaranth', 'amarena cherries', 'amaretti cookies', 'amber', 'anaheim chile', 'anardana', 'anchovies', 'anise', 'apple', 'apricot', 'arhar', 'artichoke', 'asafoetida', 'asian herb', 'asparagus', 'avocado', 'bacon', 'bagel', 'baguette', 'baking powder', 'balsamico', 'bamboo', 'banana', 'banh', 'barbecue sauce', 'barley', 'basil', 'bass', 'bay leaves', 'bean', 'bechamel', 'beef', 'beer', 'beetroot', 'biscuits', 'bisquick', 'bitters', 'blood orange', 'blueberry', 'bone', 'bonito', 'bordelaise sauce', 'bouillon', 'bread', 'broccoli', 'broth', 'butter', 'buttermilk', 'cabbage', 'cactus', 'calamari', 'candied', 'candlenuts', 'candy', 'canola oil', 'cardamon', 'carrot', 'catfish', 'caviar', 'celery', 'cereal', 'cheese', 'chicken', 'chile', 'chili', 'chili sauce', 'chipotle', 'chips', 'chive', 'chocolate', 'chocolate milk', 'ciabatta', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'codfish', 'coffee', 'cola', 'condensed cream', 'cookie', 'coriander', 'corn', 'corn starch', 'crab legs', 'cranberry', 'crepes', 'cucumber', 'cumin', 'curry', 'dill', 'dough', 'dragon fruit', 'dressing', 'duck', 'dumplings', 'eel ', 'egg', 'eggplant', 'falafel', 'fat', 'feta', 'fig', 'filet', 'fish', 'flax', 'flour', 'fries', 'garlic', 'gelatin', 'ginger', 'gnocchi', 'goat', 'goose', 'grape', 'habanero', 'ham', 'hazelnuts', 'heavy cream', 'honey', 'horse', 'hot dog', 'hot sauce', 'ice cream', 'jalapeno', 'jasmin', 'jelly', 'jerk', 'ketchup', 'kiwi', 'kung pao sauce', 'lamb', 'lasagna', 'lasagne', 'lavender', 'lemon', 'lettuce', 'lime juice', 'liqueur', 'loaf', 'lobster', 'malt', 'mandarin', 'mango', 'masala', 'mayonnaise', 'meat', 'melon', 'milk', 'mint', 'miso', 'mushroom', 'mustard', 'noodle', 'nut', 'nutella', 'oil', 'olive oil', 'olives', 'onion', 'orange', 'oregano', 'oyster sauce', 'pancake', 'paprika', 'parsley', 'passata', 'pasta', 'peach', 'peas', 'pecan', 'pepper', 'pepperoni', 'pesto', 'pig', 'pineapple', 'pistachio', 'pizza', 'pork', 'potato', 'pudding', 'pumpkin', 'rabbit', 'radish', 'ragu', 'raisin', 'raspberry', 'ribs', 'rice', 'rice noodles', 'salami', 'salmon', 'salsa', 'salt', 'sardines', 'sauerkraut', 'sausage', 'scallop', 'schmaltz', 'seafood', 'seaweed', 'seed', 'sesame', 'shallot', 'sherry', 'shrimp', 'sour cream', 'soy', 'soy sauce', 'soymilk', 'spaghetti', 'spinach', 'sprout', 'steak', 'stew', 'stock', 'strawberries', 'sugar', 'sweetener', 'syrup', 'tamarind', 'tart', 'tea ', 'thyme', 'toast', 'tofu', 'tomato', 'tomato sauce', 'tortilla', 'truffle', 'tumeric', 'tuna', 'turkey', 'vanilla', 'vinegar', 'wasabi', 'water', 'wheat', 'whipping cream', 'whiskey', 'wine', 'worcester', 'yeast', 'yogurt', 'yolk', 'zest', 'zucchini']

    const_bool_real= len(terror)

    if(len(terror)==16):

        return terror

    while(True):

        terror.append(0)

        if(len(terror)==16):

            break

    terror.sort()

    Temp_cool=15

    for jel in range(0,len(terror)):

        if(terror[jel]==0):

            terror[jel] = terror[Temp_cool]

            if(terror[Temp_cool-1]!=0):

                Temp_cool -=1

        else:

            break

    predictions = model_embedding.predict(np.array([terror]))

    listvonpred = predictions[0].tolist()

    temp_zahl_klein=[]

    for i in range(1, len(searfor)+1):

        temp_zahl_klein.append(i)

    

    for iter_num in range(len(listvonpred)-1,0,-1):

        for idx in range(iter_num):

            if listvonpred[idx]<listvonpred[idx+1]:

                temp = listvonpred[idx]

                temp2 = searfor[idx]

                temp3 = temp_zahl_klein[idx]

                listvonpred[idx] = listvonpred[idx+1]

                searfor[idx] = searfor[idx+1]

                temp_zahl_klein[idx] = temp_zahl_klein[idx+1]

                listvonpred[idx+1] = temp

                searfor[idx+1] = temp2

                temp_zahl_klein[idx+1] = temp3



    terror=terror[16-const_bool_real:16]

    const_bool_real+=1

    zahl_ai_z=0

    

    randomchoiceding=[]

    for i in listvonpred:

        if(i>0.05):

            zahl_ai_z+=1

        else:

            break

    for i in range(0,zahl_ai_z):

        for j in range(0,int(listvonpred[i]*100)):

            randomchoiceding.append(temp_zahl_klein[i])

    brich_ab=True

    if(len(randomchoiceding)!=0):

        for i in range(0,400):

            r = random.choice(randomchoiceding)

            if(not (r in terror)):

                brich_ab = False

                terror.append(r)

                break

        

    if(brich_ab):

        return terror



       # print(temp_zahl_klein[i])

      #  print(terror)

    return klein(terror)

print(searfor[Z[0]-1])



fuzzy=klein(Z)

print(fuzzy)

searfor=['achiote', 'acid', 'agar', 'agave', 'alcohol', 'alfredo sauce', 'amaranth', 'amarena cherries', 'amaretti cookies', 'amber', 'anaheim chile', 'anardana', 'anchovies', 'anise', 'apple', 'apricot', 'arhar', 'artichoke', 'asafoetida', 'asian herb', 'asparagus', 'avocado', 'bacon', 'bagel', 'baguette', 'baking powder', 'balsamico', 'bamboo', 'banana', 'banh', 'barbecue sauce', 'barley', 'basil', 'bass', 'bay leaves', 'bean', 'bechamel', 'beef', 'beer', 'beetroot', 'biscuits', 'bisquick', 'bitters', 'blood orange', 'blueberry', 'bone', 'bonito', 'bordelaise sauce', 'bouillon', 'bread', 'broccoli', 'broth', 'butter', 'buttermilk', 'cabbage', 'cactus', 'calamari', 'candied', 'candlenuts', 'candy', 'canola oil', 'cardamon', 'carrot', 'catfish', 'caviar', 'celery', 'cereal', 'cheese', 'chicken', 'chile', 'chili', 'chili sauce', 'chipotle', 'chips', 'chive', 'chocolate', 'chocolate milk', 'ciabatta', 'cinnamon', 'citrus', 'cocoa', 'coconut', 'codfish', 'coffee', 'cola', 'condensed cream', 'cookie', 'coriander', 'corn', 'corn starch', 'crab legs', 'cranberry', 'crepes', 'cucumber', 'cumin', 'curry', 'dill', 'dough', 'dragon fruit', 'dressing', 'duck', 'dumplings', 'eel ', 'egg', 'eggplant', 'falafel', 'fat', 'feta', 'fig', 'filet', 'fish', 'flax', 'flour', 'fries', 'garlic', 'gelatin', 'ginger', 'gnocchi', 'goat', 'goose', 'grape', 'habanero', 'ham', 'hazelnuts', 'heavy cream', 'honey', 'horse', 'hot dog', 'hot sauce', 'ice cream', 'jalapeno', 'jasmin', 'jelly', 'jerk', 'ketchup', 'kiwi', 'kung pao sauce', 'lamb', 'lasagna', 'lasagne', 'lavender', 'lemon', 'lettuce', 'lime juice', 'liqueur', 'loaf', 'lobster', 'malt', 'mandarin', 'mango', 'masala', 'mayonnaise', 'meat', 'melon', 'milk', 'mint', 'miso', 'mushroom', 'mustard', 'noodle', 'nut', 'nutella', 'oil', 'olive oil', 'olives', 'onion', 'orange', 'oregano', 'oyster sauce', 'pancake', 'paprika', 'parsley', 'passata', 'pasta', 'peach', 'peas', 'pecan', 'pepper', 'pepperoni', 'pesto', 'pig', 'pineapple', 'pistachio', 'pizza', 'pork', 'potato', 'pudding', 'pumpkin', 'rabbit', 'radish', 'ragu', 'raisin', 'raspberry', 'ribs', 'rice', 'rice noodles', 'salami', 'salmon', 'salsa', 'salt', 'sardines', 'sauerkraut', 'sausage', 'scallop', 'schmaltz', 'seafood', 'seaweed', 'seed', 'sesame', 'shallot', 'sherry', 'shrimp', 'sour cream', 'soy', 'soy sauce', 'soymilk', 'spaghetti', 'spinach', 'sprout', 'steak', 'stew', 'stock', 'strawberries', 'sugar', 'sweetener', 'syrup', 'tamarind', 'tart', 'tea ', 'thyme', 'toast', 'tofu', 'tomato', 'tomato sauce', 'tortilla', 'truffle', 'tumeric', 'tuna', 'turkey', 'vanilla', 'vinegar', 'wasabi', 'water', 'wheat', 'whipping cream', 'whiskey', 'wine', 'worcester', 'yeast', 'yogurt', 'yolk', 'zest', 'zucchini']



for i in fuzzy:

    print(searfor[i-1])