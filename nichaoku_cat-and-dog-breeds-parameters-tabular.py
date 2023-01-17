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
import json

# add features from ratings 

with open('../input/cat-and-dog-breeds-parameters/rating.json', 'r') as f:

        ratings = json.load(f)

breed_labels = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
cat_ratings = ratings['cat_breeds']

dog_ratings = ratings['dog_breeds']

catdog_ratings = {**cat_ratings, **dog_ratings} 
df=pd.DataFrame()
i = 0

for breed in catdog_ratings.keys():

    for key in catdog_ratings[breed].keys():

        df.at[i,'breed'] = breed

        df.at[i,key] = catdog_ratings[breed][key] 

    i = i+1
breed_labels = breed_labels.join(df.set_index('breed'), on='BreedName')

breed_labels = breed_labels.drop(['Type', 'BreedName'], axis = 1)
breed_labels.to_csv('output.csv', index = False)