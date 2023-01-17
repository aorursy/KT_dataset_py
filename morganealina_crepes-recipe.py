# Most usefull notebook on Kaggle ever...



import numpy as np

import pandas as pd



ingredients = {

    'eggs': 3,

    'flour': 250, #g

    'milk': 0.5, #L

    'sugar': 50, #g

    'oil': 1, # spoon

    'salt': 1, # pinch

}



base = pd.Series(ingredients)



base
def withEggs(eggs):

  ratio = eggs / ingredients['eggs']

  return base.mul(ratio)



withEggs(4)