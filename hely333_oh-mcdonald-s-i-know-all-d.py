
import numpy as np 
import pandas as pd 
import os
data = pd.read_csv('../input/menu.csv')



data.isnull().sum()
import matplotlib.pyplot as plt
data.pivot_table('Vitamin A (% Daily Value)', 'Category').plot(kind='bar', stacked=True)

data.pivot_table('Vitamin C (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'c')

data.pivot_table('Calcium (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'r')

data.pivot_table('Iron (% Daily Value)', 'Category').plot(kind='bar', stacked=True, color = 'k')

data.pivot_table('Protein', 'Category').plot(kind='bar', stacked=True, color = 'b')
data.pivot_table('Trans Fat', 'Category').plot(kind='bar', stacked=True, color = 'c')
data.pivot_table('Sugars', 'Category').plot(kind='bar', stacked=True, color = 'g')
data.pivot_table('Cholesterol', 'Category').plot(kind='bar', stacked=True, color = 'c')
data.pivot_table('Calories', 'Category').plot(kind='bar', stacked=True, color = 'c')
import seaborn as sns
cols = ['Calories','Cholesterol','Trans Fat','Sugars','Dietary Fiber']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)
plt.figure(figsize=(12,5))
plt.title("Distribution Calories")
ax = sns.distplot(data["Calories"], color = 'r')

print(data.Calories.mean())
print(data.Calories.median())
plt.figure(figsize=(12,5))
plt.title("Distribution Sugars")
ax = sns.distplot(data["Sugars"], color = 'c')


print(data.Sugars.mean())
print(data.Sugars.median())
def plot(grouped):
    item = grouped["Item"].sum()
    item_list = item.sort_index()
    item_list = item_list[-20:]
    plt.figure(figsize=(9,10))
    graph = sns.barplot(item_list.index,item_list.values)
    labels = [aj.get_text()[-40:] for aj in graph.get_yticklabels()]
    graph.set_yticklabels(labels)
sugar = data.groupby(data["Sugars"])
plot(sugar)
vitaminC = data.groupby(data["Vitamin C (% Daily Value)"])
plot(vitaminC)
vitaminA = data.groupby(data["Vitamin A (% Daily Value)"])
plot(vitaminA)
protein = data.groupby(data["Protein"])
plot(protein)
cholesterol = data.groupby(data["Cholesterol"])
plot(cholesterol)
fats = data.groupby(data["Trans Fat"])
plot(fats)
calories = data.groupby(data["Calories"])
plot(calories)