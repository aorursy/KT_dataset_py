import pandas as pd

import os

datadir_ext = '/kaggle/input/dsb2019-external-data/'

media_sequence = pd.read_csv(os.path.join(datadir_ext, 'media_sequence.csv'))
title_crystal = [

    'Crystal Caves - Level 1',

    'Chow Time',

    'Balancing Act',

    'Chicken Balancer (Activity)', 

    'Lifting Heavy Things', 

    'Crystal Caves - Level 2',

    'Honey Cake',

    'Happy Camel',

    'Cart Balancer (Assessment)',

    'Leaf Leader',

     'Crystal Caves - Level 3',

    'Heavy, Heavier, Heaviest',

    'Pan Balance',

    'Egg Dropper (Activity)',

    'Chest Sorter (Assessment)'

]



title_treetop = [

    'Tree Top City - Level 1',

    'Ordering Spheres',

    'All Star Sorting',

    'Costume Box',

    'Fireworks (Activity)',

    '12 Monkeys',

    'Tree Top City - Level 2',

    'Flower Waterer (Activity)',

    "Pirate's Tale",

    'Mushroom Sorter (Assessment)',

    'Air Show',

    'Treasure Map',

    'Tree Top City - Level 3',

    'Crystals Rule',

    'Rulers',

    'Bug Measurer (Activity)',

    'Bird Measurer (Assessment)'

]



title_magma = [

    'Magma Peak - Level 1',

    'Sandcastle Builder (Activity)',

    'Slop Problem',

    'Scrub-A-Dub',

    'Watering Hole (Activity)',

    'Magma Peak - Level 2',

    'Dino Drink',

    'Bubble Bath',

    'Bottle Filler (Activity)',

    'Dino Dive',

    'Cauldron Filler (Assessment)'

]
media_sequence['world'] = ['NONE'] + len(title_treetop) * ['TREETOPCITY'] + len(title_magma) * ['MAGMAPEAK'] + len(title_crystal) * ['CRYSTALCAVES']

media_sequence
media_sequence.to_csv('media_sequence_world.csv', index=False)