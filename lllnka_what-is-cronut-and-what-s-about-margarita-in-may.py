# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from plotnine import *

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/seasonality-values-d2b9a3.csv')
dataset.sample(10)
dataset['year'], dataset['week'] = dataset.week_id.str.split('-').str
dataset[['week','year']] = dataset[['week', 'year']].astype(int)
class_map = {
 'anise': 'spice',
 'apple': 'fruit',
 'apple-au': 'fruit',
 'apple-cider': 'drink',
 'apple-pie': 'sweets',
 'apple-ru': 'fruit',
 'apricot': 'fruit',
 'asparagus': 'vegetable',
 'asparagus-de': 'vegetable',
 'asparagus-jp': 'vegetable',
 'avocado': 'vegetable',
 'baked-beans': 'meal',
 'baked-chicken': 'meal',
 'banana-bread': 'meal',
 'barbecue-sauce': 'meal',
 'beef': 'meat',
 'beef-steak': 'meal',
 'beer': 'alcoholic-drink',
 'beet': 'vegetable',
 'blackberry': 'berry',
 'blueberry': 'berry',
 'boysenberry': 'berry',
 'bread': 'meal',
 'broccoli': 'vegetable',
 'brussel-sprouts': 'vegetable',
 'burrito': 'meal',
 'cabbage': 'vegetable',
 'caipirinha': 'alcoholic-drink',
 'cardamom': 'spice',
 'carrot': 'vegetable',
 'cauliflower': 'vegetable',
 'celeriac': 'vegetable',
 'champagne': 'alcoholic-drink',
 'chanterelle': 'mushroom',
 'charoset': 'meal',
 'cheeseburger': 'meal',
 'cherry-tomato': 'vegetable',
 'chia': 'seed',
 'chili': 'herb',
 'chili-con-carne': 'meal',
 'chinese-cabbage': 'vegetable',
 'chinese-water-chestnut': 'vegetable',
 'chives': 'vegetable',
 'chocolate': 'sweets',
 'chocolate-cake': 'sweets',
 'chocolate-chip-cookies': 'sweets',
 'chocolate-mousse': 'sweets',
 'chokecherry': 'berry',
 'cinnamon': 'spice',
 'coconut': 'fruit',
 'coffee': 'drink',
 'cold-brew-coffee': 'drink',
 'coriander': 'spice',
 'corn-salad': 'meal',
 'cornbread': 'meal',
 'cosmopolitan': 'alcoholic-drink',
 'cranberry': 'berry',
 'cronut': 'sweets',
 'cucumber': 'vegetable',
 'daiquiri': 'alcoholic-drink',
 'date': 'other',
 'diet': 'other',
 'dill': 'herb',
 'donut': 'sweets',
 'dumpling': 'meal',
 'easter-egg': 'meal',
 'eggplant': 'vegetable',
 'elderberry': 'berry',
 'empanada': 'meal',
 'endive': 'vegetable',
 'energy-drink': 'drink',
 'feijoa': 'fruit',
 'fennel': 'herb',
 'fig': 'fruit',
 'frozen-yogurt': 'sweets',
 'fruit-salad': 'sweets',
 'garden-tomato': 'vegetable',
 'garlic': 'vegetable',
 'gefilte-fish': 'meal',
 'gimlet': 'alcoholic-drink',
 'ginger': 'spice',
 'gooseberry': 'berry',
 'grape': 'fruit',
 'grapefruit': 'fruit',
 'grasshopper-cocktail': 'alcoholic-drink',
 'hamburger': 'meal',
 'honey': 'sweets',
 'horseradish': 'vegetable',
 'hot-chocolate': 'drink',
 'hot-dog': 'meal',
 'huckleberry': 'berry',
 'ice-cream': 'sweets',
 'iced-tea': 'drink',
 'kale': 'vegetable',
 'kale-de': 'vegetable',
 'kale-jp': 'vegetable',
 'kimchi': 'meal',
 'kohlrabi': 'vegetable',
 'kumquat': 'fruit',
 'lamb': 'meat',
 'lasagna': 'meal',
 'leek': 'vegetable',
 'lemon': 'fruit',
 'lemon-balm': 'herb',
 'lobster': 'seafood',
 'long-island-iced-tea': 'alcoholic-drink',
 'lychee': 'fruit',
 'macaron': 'sweets',
 'mai-tai': 'alcoholic-drink',
 'maitake': 'mushroom',
 'mandarin-orange': 'fruit',
 'mango': 'fruit',
 'manhattan-cocktail': 'alcoholic-drink',
 'margarita': 'alcoholic-drink',
 'marshmallow': 'sweets',
 'martini': 'alcoholic-drink',
 'marzipan': 'sweets',
 'matzah-ball': 'meal',
 'meatball': 'meal',
 'meatloaf': 'meal',
 'meringue': 'sweets',
 'microgreen': 'vegetable',
 'mimosa': 'alcoholic-drink',
 'mint-julep': 'alcoholic-drink',
 'mojito': 'alcoholic-drink',
 'moscow-mule': 'alcoholic-drink',
 'mulberry': 'berry',
 'mushroom': 'mushroom',
 'nachos': 'meal',
 'napa-cabbage': 'vegetable',
 'nectarine': 'fruit',
 'negroni': 'alcoholic-drink',
 'nougat': 'sweets',
 'nutmeg': 'spice',
 'okra': 'vegetable',
 'old-fashioned': 'alcoholic-drink',
 'onion': 'vegetable',
 'orange': 'fruit',
 'parsnip': 'vegetable',
 'pasta-salad': 'meal',
 'peach': 'fruit',
 'pear': 'fruit',
 'peppermint': 'herb',
 'persimmon': 'fruit',
 'pho': 'meal',
 'pie': 'sweets',
 'pina-colada': 'alcoholic-drink',
 'pineapple': 'fruit',
 'pizza': 'meal',
 'plum': 'fruit',
 'pomegranate': 'fruit',
 'popcorn': 'meal',
 'pork': 'meat',
 'pork-chops': 'meal',
 'pot-pie': 'meal',
 'potato': 'vegetable',
 'potato-gratin': 'meal',
 'pummelo': 'fruit',
 'pumpkin': 'vegetable',
 'pumpkin-spice-latte': 'drink',
 'quince': 'fruit',
 'quinoa': 'seed',
 'radish': 'vegetable',
 'ravioli': 'meal',
 'red-wine': 'alcoholic-drink',
 'redcurrant': 'berry',
 'rhubarb': 'vegetable',
 'risotto': 'meal',
 'rosemary': 'herb',
 'salad': 'meal',
 'salmon': 'seafood',
 'sex-on-the-beach': 'alcoholic-drink',
 'shallot': 'vegetable',
 'shirley-temple': 'alcoholic-drink',
 'shrimp': 'seafood',
 'sidecar': 'other',
 'smoothie': 'drink',
 'soup': 'meal',
 'sour-cherry': 'fruit',
 'spaghetti': 'meal',
 'spinach': 'vegetable',
 'star-anise': 'spice',
 'stew': 'meal',
 'strawberry': 'berry',
 'sushi': 'meal',
 'sweet-cherry': 'fruit',
 'sweet-potato': 'vegetable',
 'swiss-chard': 'vegetable',
 'taco': 'meal',
 'tamale': 'meal',
 'tea': 'drink',
 'tequila-sunrise': 'alcoholic-drink',
 'tom-collins': 'alcoholic-drink',
 'turkey': 'meat',
 'turmeric': 'spice',
 'turnip': 'vegetable',
 'watermelon': 'fruit',
 'whiskey-sour': 'alcoholic-drink',
 'white-russian': 'alcoholic-drink',
 'white-wine': 'alcoholic-drink',
 'wild-leek': 'herb'
}
dataset['foodclass'] = dataset.id.apply(lambda x: class_map.get(x, 'other'))
max_week = int(dataset.week.max())
min_week = int(dataset.week.min())
max_val = int(dataset.value.max())

def get_names_by_class(food_class):
    return [name for name, klass in class_map.items() if klass == food_class]


def plot_year_circle(dataset, name, years, max_week=max_week, min_week=min_week, max_val=max_val, fig_size=8):
    data = dataset[(dataset.id == name) & (dataset.year.isin(years))]
    weeks = data.week.values / max_week
    values = data.value.values / max_val
    color_map = dict(zip(years, sns.color_palette('muted', n_colors=len(years))))
    
    theta = 2 * np.pi * weeks
    area = 200 * values**2
    colors = data.year.apply(color_map.get).values

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(theta, values, c=colors, s=area, cmap='hsv', alpha=0.75)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    angles = [360./26.*i for i in range(min_week, int(max_week/2)+1)]

    labels = [i*2 for i in range(min_week, int(max_week/2)+1)]
    ax.set_thetagrids(angles, labels)
    ax.set_title(f"Year Circle for {name} in {', '.join(map(str,years))}".title(), fontsize=16)

    marks = []
    labels = []
    for year, color in color_map.items():
        marks.append(mpatches.Circle((0,0), radius=1, fc=color))
        labels.append(year)
    plt.legend(marks, years)
    
    ax.legend()
    plt.draw()

    
def plot_weekly_avg(ids, name, accented=None, accented_size=1.7, size=0.7):
    subset = dataset[dataset.id.isin(ids)]
    weeklyAvg = subset.groupby(['year', 'id']).value.mean().to_frame().reset_index()
    line_size = size
    if accented:
        line_size = weeklyAvg.id.map(lambda x: accented_size if x == accented else size)
    (
        ggplot(weeklyAvg, aes('year', 'value', color='id'))
        + geom_point()
        + geom_line(size=line_size)
        + ggtitle('Weekly Average for ' + name)
    ).draw()
    #weeklyAverage.pivot('year', 'id', 'value').plot(figsize=(12,8)) ## plot with pandas


def plot_food_set(names, years=(2016, 2015, 2014)):
    foodtrend = dataset[dataset.id.isin(names) & dataset.year.isin(years)]

    (
        ggplot(foodtrend, aes('week', 'value', color='id'))
         + geom_point()
         + ggtitle(f"{', '.join(names)} in {', '.join(map(str, years))}".title())
    ).draw()
df = dataset[dataset.year.isin([2014, 2015, 2016])].groupby(['foodclass', 'week']).mean().reset_index()

fgrid = sns.FacetGrid(df, col="foodclass", col_wrap=5, hue='foodclass')
fgrid.map(plt.scatter, "week", "value")
plot_year_circle(dataset, name='diet', years=(2014, 2015, 2016))
plot_year_circle(dataset, name='chia', years=(2014, 2015, 2016))
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
df = (dataset[dataset.id.isin(['avocado', 'chia', 'diet', 'kale', 'beef', 'quinoa', 'tea']) & dataset.year.isin([2016, 2015, 2014])]
        .pivot(index='week_id', columns='id', values='value'))
sns.heatmap(df.corr(), cmap="YlGnBu", square=True, annot=True, linewidths=.5, ax=ax)
plot_year_circle(dataset, name='barbecue-sauce', years=(2014, 2015, 2016))
plot_year_circle(dataset, name='beef', years=(2014, 2015, 2016))
plot_year_circle(dataset, name='margarita', years=(2014, 2015, 2016))
plot_year_circle(dataset, name='watermelon', years=(2014, 2015, 2016))
df = dataset[dataset.year.isin([2014, 2015, 2016])]
fgrid = sns.FacetGrid(df, col="foodclass", col_wrap=4)
fgrid.map(sns.boxplot, "year", "value", color=sns.color_palette('pastel')[0])
seeds = get_names_by_class('seed')
plot_weekly_avg(seeds, 'Seeds vs Chia', accented='chia')
spices = ['cinnamon', 'coriander', 'nutmeg', 'turmeric', 'chili']
plot_weekly_avg(spices, 'Spices vs Cinnamon', accented='cinnamon')
#['broccoli', 'avocado', 'asparagus', 'brussel-sprouts', 'endive', 'microgreen', 'radish', 'beet', 'cabbage', 'carrot', 'cauliflower', 'chives', 'eggplant', 'onion', 'garlic']
veggies = ['broccoli', 'avocado', 'beet', 'potato', 'kale', 'microgreen'] #get_names_by_class('vegetable')
plot_weekly_avg(veggies, 'Selected Veggies vs Potato', 'potato')
drinks = ['coffee', 'tea', 'hot-chocolate', 'energy-drink', 'smoothie']
plot_weekly_avg(drinks, 'Drinks vs Tea', 'tea')
fruits = ['watermelon', 'sweet-cherry', 'lychee', 'nectarine', 'apple', 'feijoa'] #get_names_by_class('fruit')
plot_weekly_avg(fruits, 'Fruits vs Apple', 'apple')
meals = ['lasagna', 'popcorn', 'pizza', 'sushi'] #get_names_by_class('meal')
plot_weekly_avg(meals, 'Meals vs Pizza', 'pizza')
seafood = get_names_by_class('seafood')
plot_weekly_avg(seafood, 'Seafood vs Salmon', 'salmon', 1.55)
sweets = ['apple-pie', 'chocolate-cake', 'cronut', 'donut', 'macaron'] #get_names_by_class('sweets')
plot_weekly_avg(sweets, 'Sweets vs Apple-Pie', 'apple-pie')
meats = get_names_by_class('meat')
plot_weekly_avg(meats, 'Meats vs Turkey', 'turkey')
plot_food_set(['ice-cream', 'hot-chocolate'])
plot_food_set(['salmon', 'beef'])
plot_food_set(['red-wine', 'white-wine'])
plot_food_set(['mojito', 'daiquiri'])