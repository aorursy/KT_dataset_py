import re

import nltk

import spacy

import plotly as plt

import scipy as sp

import pandas as pd

import en_core_web_sm

import en_core_web_lg

from re import search

from ipywidgets import *

from spacy.matcher import PhraseMatcher

from IPython.display import display, HTML
nlp = en_core_web_sm.load()
data = pd.DataFrame(

    ["""RASPBERRY FILLING (INVERT SUGAR, CORN SYRUP, SUGAR, RASPBERRY PUREE, GLYCERIN, 

    MALTODEXTRIN, MODIFIED CORN STARCH, RASPBERRY JUICE CONCENTRATE, SODIUM ALGINATE, 

    METHYLCELLULOSE, VEGETABLE JUICE FOR COLOR (RADISH, CARROT, APPLE, BLACKCURRANT, 

    HIBISCUS CONCENTRATES), MONOCALCIUM PHOSPHATE, XANTHAN GUM, DICALCIUM PHOSPHATE, 

    CITRIC ACID, NATURAL FLAVORS, MALIC ACID), WHOLE GRAIN ROLLED OATS, 

    WHOLE GRAIN WHEAT FLOUR, ENRICHED WHEAT FLOUR (BLEACHED WHEAT FLOUR, 

    MALTED BARLEY FLOUR, NIACIN, REDUCED IRON, THIAMIN MONONITRATE, RIBOFLAVIN, 

    FOLIC ACID), VEGETABLE OIL BLEND (CANOLA, PALM, PALM KERNEL), INVERT SUGAR, 

    SUGAR, GLYCERIN, CONTAINS LESS THAN 2% OF THE FOLLOWING: WHEY, SOLUBLE CORN FIBER, 

    CALCIUM CARBONATE HONEY, WHEAT BRAN, SALT, POTASSIUM BICARBONATE (LEAVENING), 

    SORBITAN MONOSTEARATE, VITAL WHEAT GLUTEN, CORN STARCH, XANTHAN GUM, REDUCED IRON, 

    NIACINAMIDE, PYRIDOXINE HYDROCHLORIDE (VITAMIN B6), DICALCIUM PHOSPHATE, ZINC OXIDE, 

    VITAMIN A PALMITATE, FOLIC ACID, THIAMIN HYDROCHLORIDE (VITAMIN B1), RIBOFLAVIN 

    (VITAMIN B2), CYANOCOBALAMIN (VITAMIN B12), NATURAL FLAVOR, MOLASSES,]""", 

     """MILK, CREAM, SKIM MILK, SUGAR, BLUE FROSTING (CORN SYRUP, 

     HIGH FRUCTOSE CORN SYRUP, WATER, STABILIZERS (FOOD STARCH-MODIFIED, CELLULOSE GUM, 

     DEXTROSE, CARRAGEENAN, GUM ARABIC, CITRIC ACID, TRICALCIUM PHOSPHATE, 

     SILICON DIOXIDE), TITANIUM DIOXIDE FOR COLOR, ARTIFICIAL FLAVOR, CITRIC ACID, 

     BLUE #1), CORN SYRUP, SEQUIN CANDY (SUGAR, CORN STARCH, RICE FLOUR, 

     PARTIALLY HYDROGENATED VEGETABLE OIL (SOYBEAN, COTTONSEED), GUM ARABIC, XANTHAN GUM, 

     CONFECTIONER GLAZE, NATURAL AND ARTIFICIAL FLAVORS, MONO & DIGLYCERIDES, 

     POLYSORBATE 60, TITANIUM DIOXIDE FOR COLOR, YELLOW 5, YELLOW 6, BLUE 1, RED 3, 

     BLUE 1 LAKE), CONTAINS 1% OR LESS OF THE FOLLOWING: NATURAL AND ARTIFICIAL FLAVORS, 

     CAROB BEAN GUM, GUAR GUM, MONO & DIGLYCERIDES, CARRAGEENAN CELLULOSE GEL, 

     CELLULOSE GUM,""", 

     """CREAM, SKIM MILK, LIQUID SUGAR, SEA SALT CARAMEL (CORN SYRUP, WATER, 

     SWEETENED CONDENSED SKIM MILK (SUGAR, WATER, NONFAT MILK SOLIDS), BUTTER (CREAM, 

     SALT), SEA SALT, CARRAGEENAN SODIUM BICARBONATE), MILK, CARAMEL BASE (CORN SYRUP, 

     SUGAR, WATER, NONFAT MILK SOLIDS, BUTTER (CREAM, SALT), SALT), 

     CARAMEL SEA SALT TRUFFLES (SUGAR, COCONUT OIL, CORN SYRUP, SWEETENED CONDENSED MILK 

     (MILK, SUGAR), CREAM, NONFAT MILK, MILK, WATER, COCOA (PROCESSED WITH ALKALI), 

     BUTTER (CREAM, SALT), SEA SALT, SOY LECITHIN, NATURAL FLAVORS), EGG YOLK, 

     CARAMEL COLOR,"""], 

    columns = ["ingredients"])
data["tokenized"] = [nlp(text) for text in data.ingredients]

data
#This is a function to extract 'root ingredients' from ingredient lists

#A 'root ingredient' is defined as a basic component of an item

#(similar to a factor in mathematics)



def transform(data, index):

    columns = ["ingredient", "subingredient_1", "subingredient_2", "root_ingredient"]

    new_data = pd.DataFrame(columns=columns)

    ingredient = ''

    subingredient_1 = ''

    subingredient_2 = ''

    paren_count = 0

    to_append = pd.DataFrame(columns=columns)

    for token in data:

        if str(token) ==  '(':

            paren_count = paren_count + 1

        elif str(token) ==  ')':

            paren_count = paren_count - 1



        if str(token) not in ['(', ')', ',', ':']:

            if paren_count == 0:

                ingredient = ingredient + ' ' + str(token)

            elif paren_count == 1:   

                subingredient_1 = subingredient_1 + ' ' + str(token)

            elif paren_count == 2:

                subingredient_2 = subingredient_2 + ' ' + str(token)        



        elif str(token) in [',', ':']:

            if (ingredient != '') | (subingredient_1 != '') | (subingredient_2 != ''):

                new_data=new_data.append(to_append)

                to_append = pd.DataFrame({'ingredient': ingredient.strip(), 

                                          'subingredient_1': subingredient_1.strip(),

                                          'subingredient_2': subingredient_2.strip()},

                                         index=[index])

            if paren_count == 0:

                ingredient = ''

                subingredient_1 = ''

                subingredient_2 = ''

            elif paren_count == 1:

                subingredient_1 = ''

                subingredient_2 = ''

            elif paren_count == 2:

                subingredient_2 = ''

    return new_data
new_data = pd.DataFrame()



for i in [0, 1, 2]:

    new_data = new_data.append(transform(data.tokenized[i], i))

    

new_data
def get_root(new_data):

    new_data.root_ingredient = new_data.subingredient_2

    new_data.root_ingredient[new_data.root_ingredient == ''] = new_data[new_data.root_ingredient == ''].subingredient_1

    new_data.root_ingredient[new_data.root_ingredient == ''] = new_data[new_data.root_ingredient == ''].ingredient

    return new_data.root_ingredient
i_data = get_root(new_data)

i_data
def clean(i_data):

    booleans = [[False if search("CONTAIN", x) != None else True for x in i_data],

               [False if search("WITH ALKALI", x) != None else True for x in i_data]]

    indexes = list(map(all, zip(*booleans)))

    fin_data = i_data[indexes]

    return fin_data     
fin_data = clean(i_data)

fin_data
print('Here are the items in list 1: \n' + 

      str(list(fin_data[fin_data.index==0].unique())), end = ' ')
print('Here are the items in list 2: \n' + 

      str(list(fin_data[fin_data.index==1].unique())), end = ' ')
print('Here are the items in list 3: \n' + 

      str(list(fin_data[fin_data.index==2].unique())), end = ' ')
synonyms = pd.DataFrame(columns=["syn1", "syn2"])

ingredient = ''

subingredient = ''



for i in range(1, len(new_data.ingredient)-1):

    if (ingredient != new_data.ingredient.iloc[i]) & (new_data.ingredient.iloc[i+1] != new_data.ingredient.iloc[i]):

        if new_data.subingredient_1.iloc[i] != '':

            synonyms = synonyms.append({"syn1": new_data.ingredient.iloc[i],

                                       "syn2": new_data.subingredient_1.iloc[i]},

                                      ignore_index=[1])

    if (subingredient != new_data.subingredient_1.iloc[i]) & (new_data.subingredient_1.iloc[i+1] != new_data.subingredient_1.iloc[i]):

        if new_data.subingredient_2.iloc[i] != '':

            synonyms = synonyms.append({"syn1": new_data.subingredient_1.iloc[i],

                                        "syn2": new_data.subingredient_2.iloc[i]},

                                       ignore_index=[1])

    ingredient = new_data.ingredient.iloc[i]

    subingredient = new_data.subingredient_1.iloc[i]

    

synonyms
fin_toke = [nlp(text) for text in fin_data]



columns = ["descriptor", "type"]

descriptors = pd.DataFrame(columns=columns)

for i in range(0, len(fin_toke)):

    for token in fin_toke[i]:

        print('%r (%s)' % (str(token), token.dep_))

        descriptors = descriptors.append({"descriptor": token,

                                          "type": token.dep_}, ignore_index=True)