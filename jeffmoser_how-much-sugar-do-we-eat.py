
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
world_sugars = world_food_facts[world_food_facts.sugars_100g.notnull()]

def return_sugars(country):
    # note that .tolist() isn't needed
    return world_sugars[world_sugars.countries == country].sugars_100g
    
# Get list of sugars per 100g for some countries
fr_sugars = return_sugars('france') + return_sugars('en:fr')
za_sugars = return_sugars('south africa')
uk_sugars = return_sugars('united kingdom') + return_sugars('en:gb')
us_sugars = return_sugars('united states') + return_sugars('en:us') + return_sugars('us')
sp_sugars = return_sugars('spain') + return_sugars('españa') + return_sugars('en:es')
nd_sugars = return_sugars('netherlands') + return_sugars('holland')
au_sugars = return_sugars('australia') + return_sugars('en:au')
cn_sugars = return_sugars('canada') + return_sugars('en:cn')
de_sugars = return_sugars('germany')


# You can call mean directly on numpy
np.mean(de_sugars)
# or call it directly on the series/column:
de_sugars.mean()
# For reference, here's the original mean function you wrote
def mean(l):
    return float(sum(l)) / len(l)
mean(de_sugars)
# You can import it as a different name:
from numpy import mean as mean2
mean2(de_sugars)
# or just simply:
from numpy import mean
mean(de_sugars)
