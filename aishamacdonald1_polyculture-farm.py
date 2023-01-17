# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

# import matplotlib.pyplot as plt

# import matplotlib.image as mpimg

from bqplot import *

# from bqplot import Pie, Figure

# import numpy as np

import string

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# plt.style.use('classic')

# %matplotlib inline

# !pip install bqplot
# !jupyter nbextension enable --py --sys-prefix bqplot
# clean data in case I ran it before

results = []

table_results = []

historical_data = []



# Date

year = datetime.datetime(2001, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = 100

ducks = 100

sheep = 10

bulls = 1

pigs = 16

dogs = 1



# Fruit and plants

apple_trees = 36

orange_trees = 40

plum_trees = 32

nectarine_trees = 28

pear_trees = 23



# Problem animals

coyotes = 3

gophers = 0

birds = 2000

aphids_in_millions = 10



# Prices

apple_price = 0.89

orange_price = 0.99

plum_price = 0.65

nectarine_price = 0.73 

pear_price = 0.50

egg_price = 0.50

eggs_per_bird = 330



# For calculation

good_apple_percentage = 0.3

good_orange_percentage = 0.5

good_plum_percentage = 0.2

good_nectarine_percentage = 0.3

good_pear_percentage = 0.6



apples_per_tree = 200

oranges_per_tree = 300

plums_per_tree = 100

nectarines_per_tree = 200

pears_per_tree = 300



# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees



good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

# total_apples = round(total_apples,0) 

# total_oranges = round(total_oranges,0) 

# total_plums = round(total_plums,0) 

# total_nectarines = round(total_nectarines,0) 

# total_pears = round(total_pears,0) 

total_eggs = chickens * eggs_per_bird



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 





bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

column_names = ['year','chickens','egg_price','eggs_per_bird','total_eggs','ducks','sheep','bulls','pigs','dogs','apple_trees','orange_trees','plum_trees','nectarine_trees','pear_trees','coyotes','gophers','birds','aphids_in_millions','apple_price','orange_price','plum_price','nectarine_price','pear_price','good_apple_percentage','good_orange_percentage','good_plum_percentage','good_nectarine_percentage','good_pear_percentage','apples_per_tree','oranges_per_tree','plums_per_tree','nectarines_per_tree','pears_per_tree','total_apples','total_oranges','total_plums','total_nectarines','total_pears','good_apples','good_oranges','good_plums','good_nectarines','good_pears','bad_apples','bad_oranges','bad_plums','bad_nectarines','bad_pears']



historical_data = [[year,chickens,egg_price,eggs_per_bird,total_eggs,ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears]]



print("Some highlights of the year")

print("We have apple trees")

print("We had " + str(total_apples) + " apples from our " + str(apple_trees) + " trees")

print("We had " + str(good_apples) + " good apples")

print("We had " + str(bad_apples) + " bad apples")



print("We sold our apples on average for $" + str(apple_price))



money_made = apple_price * good_apples

print("in year 1 we made $" + str(money_made) + " from the good apples sold") 



pie_data_df = pd.DataFrame(historical_data, columns = ['year','chickens','egg_price','eggs_per_bird','total_eggs','ducks','sheep','bulls','pigs','dogs','apple_trees','orange_trees','plum_trees','nectarine_trees','pear_trees','coyotes','gophers','birds','aphids_in_millions','apple_price','orange_price','plum_price','nectarine_price','pear_price','good_apple_percentage','good_orange_percentage','good_plum_percentage','good_nectarine_percentage','good_pear_percentage','apples_per_tree','oranges_per_tree','plums_per_tree','nectarines_per_tree','pears_per_tree','total_apples','total_oranges','total_plums','total_nectarines','total_pears','good_apples','good_oranges','good_plums','good_nectarines','good_pears','bad_apples','bad_oranges','bad_plums','bad_nectarines','bad_pears'])

fruit_df = pie_data_df[['orange_trees','apple_trees','plum_trees','nectarine_trees','pear_trees']]

farm_animals_df = pie_data_df[[

    'chickens',

    'dogs',

    'ducks',

    'bulls',

    'sheep',

    'pigs',

]]

explode = (0, 0.5, 0, 0, 0)

data = fruit_df.values.tolist()

pie = Pie(sizes=data, display_labels='outside', labels=fruit_df.columns.tolist())

fig = Figure(marks=[pie], animation_duration=1000, title='Fruit Trees in our First Year')



with pie.hold_sync():

    pie.display_values = True

    pie.values_format = '.0f'

    pie.label_color = 'Green'

    pie.font_size = '16px'

    pie.font_weight = 'bold'

    pie.title = 'Number of fruit trees'

    

fig

data = farm_animals_df.values.tolist()

pie = Pie(sizes=data, display_labels='outside', labels=farm_animals_df.columns.tolist())

fig = Figure(marks=[pie], animation_duration=1000, title='Farm Animals in our First Year')

with pie.hold_sync():

    pie.display_values = True

    pie.values_format = '.0f'

    pie.label_color = 'Green'

    pie.font_size = '16px'

    pie.font_weight = 'bold'

fig
# Date

year = datetime.datetime(2002, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = 260

ducks = 100

sheep = 12

bulls = 1

pigs = 16

dogs = 4 



# Fruit and plants

apple_trees = 34

orange_trees = 39

plum_trees = 32

nectarine_trees = 27

pear_trees = 23



# Problem animals

coyotes = 5

gophers = 124

birds = 2730

aphids_in_millions = 14



# Prices

apple_price = 0.99

orange_price = 0.99

plum_price = 0.79

nectarine_price = 0.80

pear_price = 0.50

egg_price = 0.50

eggs_per_bird = 330



# For calculation

good_apple_percentage = 0.2

good_orange_percentage = 0.6

good_plum_percentage = 0.3

good_nectarine_percentage = 0.4

good_pear_percentage = 0.7



apples_per_tree = 200 - 39

oranges_per_tree = 300 - 70

plums_per_tree = 100 - 0

nectarines_per_tree = 200 - 23

pears_per_tree = 300 - 0





# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees

total_eggs = chickens * eggs_per_bird



good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

total_apples = round(total_apples,0) 

total_oranges = round(total_oranges,0) 

total_plums = round(total_plums,0) 

total_nectarines = round(total_nectarines,0) 

total_pears = round(total_pears,0) 



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 





bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

results = [year,chickens,egg_price,eggs_per_bird,total_eggs,ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears,]

historical_data.append(results)



print("Some highlights of the year")

print("We have pear trees")

print("We had " + str(total_pears) + " pears from our " + str(pear_trees) +  " trees")

print("We had " + str(good_pears) + " good pears")

print("We had " + str(bad_pears) + " bad pears")



print("We sold our pears on average for $" + str(pear_price))



money_made = pear_price * good_pears

print("in year 2 we made $" + str(money_made) + " from the good pears sold") 
# Date

year = datetime.datetime(2003, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = chickens + 40

ducks = 100

sheep = 112

bulls = 1

pigs = 16

dogs = 4 



# Fruit and plants

apple_trees = 34

orange_trees = 39

plum_trees = 32

nectarine_trees = 27

pear_trees = 23



# Problem animals

coyotes = 5

gophers = 124

birds = 2730

aphids_in_millions = 14



# Prices

apple_price = 0.99

orange_price = 0.99

plum_price = 0.79

nectarine_price = 0.80

pear_price = 0.50

egg_price = 0.50

eggs_per_bird = 330



# For calculation

# good apples are the ones the birds did not destroy

good_apple_percentage = 0.65

good_orange_percentage = 0.88

good_plum_percentage = 0.72

good_nectarine_percentage = 0.69

good_pear_percentage = 0.63



apples_per_tree = 200 - 39 + 100

oranges_per_tree = 300 - 70 + 100

plums_per_tree = 100 - 0 + 100

nectarines_per_tree = 200 - 23 + 100

pears_per_tree = 300 - 0 + 100



# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees





good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

total_apples = round(total_apples,0) 

total_oranges = round(total_oranges,0) 

total_plums = round(total_plums,0) 

total_nectarines = round(total_nectarines,0) 

total_pears = round(total_pears,0) 

total_eggs = chickens * eggs_per_bird



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 





bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

results = [year,chickens,egg_price,eggs_per_bird,total_eggs,ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears,]

historical_data.append(results)



print("Some highlights of the year")

print("We have plum trees")

print("We had " + str(total_plums) + " plums from our " + str(plum_trees) + " trees")

print("We had " + str(good_plums) + " good plums")

print("We had " + str(bad_plums) + " bad plums")



print("We sold our plum on average for $" + str(plum_price))



money_made = plum_price * good_apples

print("in year 3 we made $" + str(money_made) + " from the good plums sold") 
# Date

year = datetime.datetime(2004, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = 30

ducks = 100

sheep = 108

bulls = 1

pigs = 16

dogs = 4 



# Fruit and plants

apple_trees = 25

orange_trees = 36

plum_trees = 31

nectarine_trees = 25

pear_trees = 21



# Problem animals

coyotes = 18

gophers = 111

birds = 2949

aphids_in_millions = 11



# Prices

apple_price = 0.95

orange_price = 0.82

plum_price = 0.99

nectarine_price = 0.90

pear_price = 0.70

egg_price = 0.50

eggs_per_bird = 330



# For calculation

# good apples are the ones the birds did not destroy

good_apple_percentage = 0.42

good_orange_percentage = 0.63

good_plum_percentage = 0.27

good_nectarine_percentage = 0.42

good_pear_percentage = 0.77



apples_per_tree = 200 - 39 + 100 - 90

oranges_per_tree = 300 - 70 + 100 - 70

plums_per_tree = 100 - 0 + 100 - 120

nectarines_per_tree = 200 - 23 + 100 - 100

pears_per_tree = 300 - 0 + 100 - 40



# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees





good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

total_apples = round(total_apples,0) 

total_oranges = round(total_oranges,0) 

total_plums = round(total_plums,0) 

total_nectarines = round(total_nectarines,0) 

total_pears = round(total_pears,0) 

total_eggs = chickens * eggs_per_bird



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 





bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

results = [year,chickens,egg_price,eggs_per_bird,total_eggs,ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears,]

historical_data.append(results)



print("Some highlights of the year")

print("We have orange trees")

print("We had " + str(total_oranges) + " oranges from our " + str(orange_trees) +  " trees")

print("We had " + str(good_oranges) + " good oranges")

print("We had " + str(bad_oranges) + " bad oranges")



print("We sold our orange on average for $" + str(orange_price))



money_made = orange_price * good_oranges

print("in year 4 we made $" + str(money_made) + " from the good oranges sold") 
from ipywidgets import Layout

fig_layout = Layout(width='960px', height='800px')

node_data = [

    {'label': 'aphids', 'shape': 'rect', 'shape_attrs': {'rx': 50, 'ry': 50,'width': 80, 'height': 30}, 'relationship': 'destroy plants'},

    {'label': 'ladybugs', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'eat aphids'},

    {'label': 'fruit trees', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'make money'},

    {'label': 'owls', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'catch gophers and live in trees'},

    {'label': 'gophers', 'shape': 'rect', 'shape_attrs': {'rx': 50, 'ry': 50,'width': 80, 'height': 30}, 'relationship': 'eat plant roots'},

    {'label': 'rain', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'helps the soil'},

    {'label': 'humans', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'look after the farm and kill the coyotes'},

    {'label': 'soil', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'helps the fruit trees'},

    {'label': 'worms', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'help the soil'},  

    {'label': 'coyotes', 'shape': 'rect', 'shape_attrs': {'rx': 50, 'ry': 50,'width': 80, 'height': 30}, 'relationship': 'kill the gophers and chickens'},

    {'label': 'dogs', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'look after farm animals'},

    {'label': 'hawks', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'catch gophers'},

    {'label': 'ground cover plants', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 120, 'height': 30}, 'relationship': 'holds the water, good for the soil and keeps weeds away'},

    {'label': 'farm animals', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'the poo is good for the soil and they make money'},

    {'label': 'ducks', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'eat the snails'},

    {'label': 'snails', 'shape': 'rect', 'shape_attrs': {'rx': 50, 'ry': 50,'width': 80, 'height': 30}, 'relationship': 'ruin the trees and plants but feed the chickens and ducks'},

    {'label': 'chickens', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'makes money, good for the soil and eat snails and slugs'},

    {'label': 'snakes', 'shape': 'rect', 'shape_attrs': {'rx': 10, 'ry': 10,'width': 80, 'height': 30}, 'relationship': 'eat gophers'},

    {'label': 'wild birds', 'shape': 'rect', 'shape_attrs': {'rx': 50, 'ry': 50,'width': 80, 'height': 30}, 'relationship': 'peck into the fruit, but help to polinate the trees'},    

]



link_data = [

    {'source': 0, 'target': 2},

    {'source': 1, 'target': 0},

    {'source': 2, 'target': 6},

    {'source': 3, 'target': 4},

    {'source': 4, 'target': 2},

    {'source': 5, 'target': 7},

    {'source': 6, 'target': 2},

    {'source': 6, 'target': 13},

    {'source': 6, 'target': 9},

    {'source': 6, 'target': 10},

    {'source': 7, 'target': 2},

    {'source': 8, 'target': 7},

    {'source': 9, 'target': 4},

    {'source': 9, 'target': 16},

    {'source': 10, 'target': 13},

    {'source': 11, 'target': 4},

    {'source': 12, 'target': 7},

    {'source': 12, 'target': 7},

    {'source': 12, 'target': 12},

    {'source': 13, 'target': 7},

    {'source': 13, 'target': 6},

    {'source': 14, 'target': 15},

    {'source': 15, 'target': 16},

    {'source': 15, 'target': 14},

    {'source': 15, 'target': 2},

    {'source': 16, 'target': 7},

    {'source': 16, 'target': 6},

    {'source': 16, 'target': 15},

    {'source': 17, 'target': 4},

    {'source': 18, 'target': 2},

]

print("See if you can untangle the relationships with you your mouse")

graph = Graph(node_data=node_data, link_data=link_data, charge=-600, colors=['pink'] * 7)

tooltip = Tooltip(fields=['label', 'relationship'], formats=['', ''])

graph.tooltip = tooltip

Figure(marks=[graph], layout=fig_layout, title='Symbiotic Relationships on the farm')

# Date

year = datetime.datetime(2005, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = 150

ducks = 130

sheep = 120

bulls = 1

pigs = 28

dogs = 4 



# Fruit and plants

apple_trees = 42

orange_trees = 36

plum_trees = 31

nectarine_trees = 25

pear_trees = 35



# Problem animals

coyotes = 45

gophers = 800

birds = 3400

aphids_in_millions = 14



# Prices

apple_price = 0.85

orange_price = 0.95

plum_price = 0.76

nectarine_price = 0.95

pear_price = 0.80

egg_price = 0.60

eggs_per_bird = 330



# For calculation

# good apples are the ones the birds did not destroy

good_apple_percentage = 0.35

good_orange_percentage = 0.72

good_plum_percentage = 0.21

good_nectarine_percentage = 0.19

good_pear_percentage = 0.82



apples_per_tree = 200 - 39 + 100 - 90 + 140

oranges_per_tree = 300 - 70 + 100 - 70 + 120

plums_per_tree = 100 - 0 + 100 - 120 + 180

nectarines_per_tree = 200 - 23 + 100 - 100 + 250

pears_per_tree = 300 - 0 + 100 - 40 + 290



# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees

total_eggs = chickens * eggs_per_bird



good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

total_apples = round(total_apples,0) 

total_oranges = round(total_oranges,0) 

total_plums = round(total_plums,0) 

total_nectarines = round(total_nectarines,0) 

total_pears = round(total_pears,0) 

total_eggs = chickens * eggs_per_bird



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 





bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

results = [year,chickens,egg_price,eggs_per_bird, total_eggs, ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears,]

historical_data.append(results)



print("Some highlights of the year")

print("We have chickens on our farm")

print("We had " + str(total_eggs) + " eggs from our " + str(chickens) + " chickens")



print("We sold our eggs on average for $" + str(egg_price))



money_made = total_eggs * egg_price

print("in year 5 we made $" + str(money_made) + " from the eggs sold") 

# Date

year = datetime.datetime(2006, 12, 31).strftime('%Y-%m-%d')



# Animals

chickens = 370

ducks = 150

sheep = 130

bulls = 1

pigs = 28

dogs = 4 



# Fruit and plants

apple_trees = 42

orange_trees = 36

plum_trees = 31

nectarine_trees = 25

pear_trees = 35



# Problem animals

coyotes = 2

gophers = 5

birds = 150

aphids_in_millions = 0.04



# Prices

apple_price = 0.95

orange_price = 0.95

plum_price = 0.76

nectarine_price = 0.95

pear_price = 0.80

egg_price = 0.60

eggs_per_bird = 330



# For calculation

# good apples are the ones the birds did not destroy

good_apple_percentage = 0.85

good_orange_percentage = 0.83

good_plum_percentage = 0.91

good_nectarine_percentage = 0.79

good_pear_percentage = 0.86



apples_per_tree = 200 - 39 + 100 - 90 + 140 + 30

oranges_per_tree = 300 - 70 + 100 - 70 + 120 + 15

plums_per_tree = 100 - 0 + 100 - 120 + 180 + 30

nectarines_per_tree = 200 - 23 + 100 - 100 + 250 + 35

pears_per_tree = 300 - 0 + 100 - 40 + 290 + 20



# Calculatioms

total_apples = apples_per_tree * apple_trees

total_oranges = oranges_per_tree * orange_trees

total_plums = plums_per_tree * plum_trees

total_nectarines = nectarines_per_tree * nectarine_trees

total_pears = pears_per_tree * pear_trees

total_eggs = chickens * eggs_per_bird



good_apples = total_apples * good_apple_percentage

good_oranges = total_oranges * good_orange_percentage

good_plums = total_plums * good_plum_percentage

good_nectarines = total_nectarines * good_nectarine_percentage

good_pears = total_pears * good_pear_percentage



# Fix rounding issue

total_apples = round(total_apples,0) 

total_oranges = round(total_oranges,0) 

total_plums = round(total_plums,0) 

total_nectarines = round(total_nectarines,0) 

total_pears = round(total_pears,0) 



good_apples = round(good_apples,0) 

good_oranges = round(good_oranges,0) 

good_plums = round(good_plums,0) 

good_nectarines = round(good_nectarines,0) 

good_pears = round(good_pears,0) 



bad_apples = total_apples - good_apples

bad_oranges = total_oranges - good_oranges

bad_plums = total_plums - good_plums

bad_nectarines = total_nectarines - good_nectarines

bad_pears = total_pears - good_pears



# Make one row for one year of numbers

results = [year,chickens,egg_price,eggs_per_bird,total_eggs,ducks,sheep,bulls,pigs,dogs,apple_trees,orange_trees,plum_trees,nectarine_trees,pear_trees,coyotes,gophers,birds,aphids_in_millions,apple_price,orange_price,plum_price,nectarine_price,pear_price,good_apple_percentage,good_orange_percentage,good_plum_percentage,good_nectarine_percentage,good_pear_percentage,apples_per_tree,oranges_per_tree,plums_per_tree,nectarines_per_tree,pears_per_tree,total_apples,total_oranges,total_plums,total_nectarines,total_pears,good_apples,good_oranges,good_plums,good_nectarines,good_pears,bad_apples,bad_oranges,bad_plums,bad_nectarines,bad_pears,]

historical_data.append(results)



table_results = pd.DataFrame(historical_data, columns = ['year','chickens','egg_price','eggs_per_bird','total_eggs','ducks','sheep','bulls','pigs','dogs','apple_trees','orange_trees','plum_trees','nectarine_trees','pear_trees','coyotes','gophers','birds','aphids_in_millions','apple_price','orange_price','plum_price','nectarine_price','pear_price','good_apple_percentage','good_orange_percentage','good_plum_percentage','good_nectarine_percentage','good_pear_percentage','apples_per_tree','oranges_per_tree','plums_per_tree','nectarines_per_tree','pears_per_tree','total_apples','total_oranges','total_plums','total_nectarines','total_pears','good_apples','good_oranges','good_plums','good_nectarines','good_pears','bad_apples','bad_oranges','bad_plums','bad_nectarines','bad_pears'])



total_money_made = (apple_price * total_apples) + (orange_price * total_oranges) + (plum_price * total_plums) + (nectarine_price * total_nectarines) + (pear_price * total_pears) + (egg_price * total_eggs)



print("This year we made $" + str(apple_price * good_apples) + " on our apples")

print("This year we made $" + str(orange_price * good_oranges) + " on our oranges")

print("This year we made $" + str(plum_price * good_plums) + " on our plums")

print("This year we made $" + str(nectarine_price * good_nectarines) + " on our nectarines")

print("This year we made $" + str(pear_price * good_pears) + " on our pears")

print("This year we made $" + str(egg_price * total_eggs) + " on our eggs")



print("This year we made a total of $" + str(round(total_money_made,0)))

table_results[['year',

    'chickens',

    'ducks',

    'sheep',

    'bulls',

    'pigs',

    'dogs',

    'apple_trees',

    'orange_trees',

    'plum_trees',

    'nectarine_trees',

    'pear_trees',

    'total_apples',

    'total_oranges',

    'total_plums',

    'total_nectarines',

    'total_pears',

    'total_eggs',

]]