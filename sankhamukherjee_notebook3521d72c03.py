import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import radians



# The following option is form here ...

# Not sure what this does ..

# https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/simple-exploration-notebook-2-connect

pd.options.mode.chained_assignment = None  # default='warn'



train = pd.read_json("../input/train.json")

test  = pd.read_json("../input/test.json")

print("Train Rows : ", train.shape[0])

print("Test Rows : ", test.shape[0])
train.head()
predictor = 'interest_level'

numericCols = ['bathrooms', 'bedrooms', 'price']

pictureCols = ['photos']

mapColumns  = ['latitude', 'longitude']

descriptions = ['features', 'description'] # basically we need to combine these two
i = 0

allFeatures = set([])

for fs in test.features.values:

    nonVals = [f.lower() for f in fs if '**' not in f]

    for f in fs:

        if '*' in f:

            nonVals += [m.lower().strip() for m in f.replace('**', '').strip().split('*')]

            

    allFeatures = allFeatures.union( set(nonVals) )

    

print('done')
allFeatures = sorted((allFeatures))

allFeatures
import re



groupings = {

    '(blks?|mins?) *?to' : 'toPlace',

    'back ?yard': 'backyard',

    'dishwasher': 'dishwasher',

    'doorman'   : 'doorman',

    '(fitness|welness)' : 'fitness',

    

}



for feature in allFeatures:

    foundMatch = False

    for g in groupings:

        if re.search(g, feature):

            foundMatch = True

            #print(g.rjust(20),'-->' ,feature)

    if not foundMatch:

        print(feature)

        

            