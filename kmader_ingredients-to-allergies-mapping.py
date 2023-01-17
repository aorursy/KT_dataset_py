common_allergens = {

    'cows milk': {'Cheese', 'Butter', 'Margarine', 'Yogurt', 'Cream', 'Ice cream'},

    'eggs': {' egg'},

    'tree nuts': {'Brazil nut', 'Almond', 'Cashew', 'Macadamia nut', 'Pistachio','Pine nut','Walnut'},

    'peanuts': {'peanut'},

    'shellfish': {'Shrimp','Prawn','Crayfish', 'Lobster', 'Squid', 'Scallops'},

    'wheat': {'flour', 'wheat', 'pasta', 'noodle', 'bread', 'crust'},

    'soy': {'soy', 'tofu'},

    'fish': {'fish', 'seafood'}

}
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import pandas as pd

from skimage.io import imread as imread

from skimage.util import montage as montage2d

from skimage.color import label2rgb

base_dir = Path('..') / 'input'
image_overview_df = pd.read_json(base_dir/ 'image_subset.json')

image_overview_df.sample(3)
from itertools import chain

image_overview_df['clean_ingredients_list'] = image_overview_df['clean_ingredients_list'].map(lambda c_list: [x.lower() for x in c_list])

all_ingredients = list(chain.from_iterable(image_overview_df['clean_ingredients_list'].values))

ingredient_list = pd.value_counts(all_ingredients)
ing_set = ingredient_list.index.tolist()

allegen_ingredients = {}

for k,v in common_allergens.items():

    allegen_ingredients[k] = []

    for c_item in v:

        for c_ing in ing_set:

            if c_item.lower() in c_ing:

                allegen_ingredients[k].append(c_ing)

for k,v in allegen_ingredients.items():

    print(k, len(v), np.random.permutation(v)[:3])
for k, c_allergens in allegen_ingredients.items():

    image_overview_df[k] = image_overview_df['clean_ingredients_list'].map(lambda c_list: len([x for x in c_list if x in c_allergens]))

    print(k, image_overview_df[k].value_counts())
c_alls = list(allegen_ingredients.keys())

fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))

ax1.bar(range(len(c_alls)), [sum(image_overview_df[k]>0) for k in c_alls])

ax1.set_xticks(range(1, 1+len(c_alls)))

ax1.set_xticks(range(len(c_alls)))

ax1.set_xticklabels(c_alls, rotation=90)
all_arr = np.stack(image_overview_df[c_alls].values, 0)

all_arr.shape
fig, ax1 = plt.subplots(1, 1, figsize=(50, 10))

ax1.imshow(all_arr.T, cmap='viridis')

ax1.set_aspect(500)

ax1.set_xlabel('Image #')

ax1.set_ylabel('Word')

ax1.set_yticks(range(len(c_alls)))

ax1.set_yticklabels(c_alls);
sample_images_df = pd.concat([image_overview_df.\

     groupby(image_overview_df[k].map(lambda x: x>0)).\

     apply(lambda x: x.sample(1)).\

     reset_index(drop=True) 

     for k in c_alls])

print(sample_images_df.shape)
fig, m_axs = plt.subplots(4, 4, figsize=(20, 20))

for c_ax, (_, c_row) in zip(m_axs.flatten(), 

                            sample_images_df.iterrows()):

    c_ax.imshow(imread(base_dir / 'subset' / c_row['image_path']))

    c_title = ', '.join([k for k in c_alls if c_row[k]>0])

    c_ax.set_title('{}\n{}'.format(c_row['title'], c_title))

    c_ax.axis('off')
image_overview_df[['image_path', 'title', 'ingredients_list']+c_alls].to_json('clean_list.json')