import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("whitegrid")
drinks = pd.read_csv("../input/drinks.csv")
drinks.info()
drinks = drinks.rename(columns={'beer_servings':"beer", 'wine_servings':"wine",\
                       'spirit_servings':"spirit",'total_litres_of_pure_alcohol':"total"})

for coluna in drinks.columns[1:4]:
    plt.figure(figsize=(14,6))
    plt.title("Total worldwide consumption of " + coluna)
    sns.barplot(x=drinks.continent, y=drinks[coluna])
most_vine = drinks.sort_values(by = "wine", ascending=False).head(20) 
plt.figure(figsize=(14,6))
plt.title("The countries that most drink wine")
sns.barplot(y=most_vine.country, x=most_vine["wine"])
plt.figure(figsize=(16,6))
plt.title("Total alcohol consumption per continent")
sns.barplot(y=drinks.total, x=drinks.continent) 

plt.figure(figsize=(13,8))
plt.title("Total alcohol consumption per continent")
sns.violinplot(x=drinks.total, y=drinks.continent)