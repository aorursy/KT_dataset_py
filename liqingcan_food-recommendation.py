%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/en.openfoodfacts.org.products.tsv','\t')
world_food_facts.countries = world_food_facts.countries.str.lower()

world_protein = world_food_facts[world_food_facts.proteins_100g.notnull()]

def return_protein(country):
    return world_protein[world_protein.countries == country][['product_name','proteins_100g']]

# Say if we want to get highest protein food in Canada
cn_protein_food = return_protein('canada')
cn_protein_food = cn_protein_food.sort_values(['proteins_100g'], ascending=[0])
top_cn_protein_food=cn_protein_food[0:10]
# top_cn_protein_food is what we want :

