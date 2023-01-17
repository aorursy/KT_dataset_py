#The product belongs to the associated country can be find by "countries_en" column also and we don't need to add extra code such as (fr:)
%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



world_food_facts = pd.read_csv('../input/FoodFacts.csv')

world_food_facts.countries_en = world_food_facts.countries_en.str.lower()

    

def mean(l):

    return float(sum(l)) / len(l)



world_sugars = world_food_facts[world_food_facts.sugars_100g.notnull()]



def return_sugars(country):

    return world_sugars[world_sugars.countries_en == country].sugars_100g.tolist()

    

# Get list of sugars per 100g for some countries

fr_sugars = return_sugars('france')

za_sugars = return_sugars('south africa')

uk_sugars = return_sugars('united kingdom')

us_sugars = return_sugars('united states')

sp_sugars = return_sugars('spain')

nd_sugars = return_sugars('netherlands')

au_sugars = return_sugars('australia')

cn_sugars = return_sugars('canada')

de_sugars = return_sugars('germany')



countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'ND', 'AU', 'CN', 'DE']

sugars_l = [mean(fr_sugars), 

            mean(za_sugars), 

            mean(uk_sugars), 

            mean(us_sugars), 

            mean(sp_sugars), 

            mean(nd_sugars),

            mean(au_sugars),

            mean(cn_sugars),

            mean(de_sugars)]

            

y_pos = np.arange(len(countries))

    

plt.bar(y_pos, sugars_l, align='center', alpha=0.5)

plt.title('Average total sugar content per 100g')

plt.xticks(y_pos, countries)

plt.ylabel('Sugar/100g')

    

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



world_food_facts = pd.read_csv('../input/FoodFacts.csv')

world_food_facts.countries_en = world_food_facts.countries_en.str.lower()

    

def mean(l):

    return float(sum(l)) / len(l)



world_sodium = world_food_facts[world_food_facts.sodium_100g.notnull()]



def return_sodium(country):

    return world_sodium[world_sodium.countries_en == country].sodium_100g.tolist()

    

# Get list of sodium per 100g for some countries

fr_sodium = return_sodium('france')

za_sodium = return_sodium('south africa')

uk_sodium = return_sodium('united kingdom')

us_sodium = return_sodium('united states') 

sp_sodium = return_sodium('spain')

ch_sodium = return_sodium('china')

nd_sodium = return_sodium('netherlands')

au_sodium = return_sodium('australia')

jp_sodium = return_sodium('japan')

de_sodium = return_sodium('germany')



countries = ['FR', 'ZA', 'UK', 'USA', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']

sodium_l = [mean(fr_sodium), 

            mean(za_sodium), 

            mean(uk_sodium), 

            mean(us_sodium), 

            mean(sp_sodium), 

            mean(ch_sodium),

            mean(nd_sodium),

            mean(au_sodium),

            mean(jp_sodium),

            mean(de_sodium)]



y_pos = np.arange(len(countries))

    

plt.bar(y_pos, sodium_l, align='center', alpha=0.5)

plt.title('Average sodium content per 100g')

plt.xticks(y_pos, countries)

plt.ylabel('Sodium/100g')

    

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



world_food_facts = pd.read_csv('../input/FoodFacts.csv')

world_food_facts.countries_en = world_food_facts.countries_en.str.lower()

    

def mean(l):

    return float(sum(l)) / len(l)



world_additives = world_food_facts[world_food_facts.additives_n.notnull()]



def return_additives(country):

    return world_additives[world_additives.countries_en == country].additives_n.tolist()

    

# Get list of additives amounts for some countries

fr_additives = return_additives('france')

za_additives = return_additives('south africa')

uk_additives = return_additives('united kingdom')

us_additives = return_additives('united states')

sp_additives = return_additives('spain')

ch_additives = return_additives('china')

nd_additives = return_additives('netherlands')

au_additives = return_additives('australia')

jp_additives = return_additives('japan')

de_additives = return_additives('germany')



countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']

additives_l = [mean(fr_additives), 

            mean(za_additives), 

            mean(uk_additives), 

            mean(us_additives), 

            mean(sp_additives), 

            mean(ch_additives),

            mean(nd_additives),

            mean(au_additives),

            mean(jp_additives),

            mean(de_additives)]



y_pos = np.arange(len(countries))

    

plt.bar(y_pos, sodium_l, align='center', alpha=0.5)

plt.title('Average amount of additives')

plt.xticks(y_pos, countries)

plt.ylabel('Amount of additives')

    

plt.show()