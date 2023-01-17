# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TRAIN = '../input/train.json'
df = pd.read_json(TRAIN)

df.head()
CUISINES = set(df['cuisine'])

INGREDIENTS = set()

for ings in df['ingredients']:

    for ing in ings:

        INGREDIENTS.add(ing)

ingredients = {ing: {cui: 0 for cui in CUISINES} for ing in INGREDIENTS}

for index, row in df.iterrows():

    cui = row['cuisine']

    for ing in row['ingredients']:

        ingredients[ing][cui] += 1

        

recipes = {}

for cui in CUISINES:

    recipes[cui] = sum(ingredients[ing][cui] for ing in INGREDIENTS)
s = sum(recipes[cui] for cui in CUISINES)
bayes = {ing: {cui: None for cui in CUISINES} for ing in INGREDIENTS}

for ing in INGREDIENTS:

    for cui in CUISINES:

        bayes[ing][cui] = ingredients[ing][cui] / recipes[cui]



    
def predict(*ings):

    ans = None

    q = 0

    for cui in CUISINES:

        #print(cui)

        p = recipes[cui] / s

        for ing in ings:

            try:

                p *= bayes[ing][cui]

            except:

                p = 0

            #print(p)

        if p > q:

            q = p

            ans = cui

    return ans, q
predict('powdered sugar')
TEST = '../input/test.json'

test = pd.read_json(TEST)

test.head()
submission = pd.DataFrame(columns = ['id', 'cuisine'])

for index, row in test.iterrows():

    #print(index)

    dic = {'id': row['id']}

    dic['cuisine'] = predict(*row['ingredients'])[0]

    submission = submission.append([dic])

submission.head()
submission.to_csv('submission.csv', index=False)