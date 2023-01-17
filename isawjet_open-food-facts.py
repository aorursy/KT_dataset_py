%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()

# 1. using groupby to group the rows by the country
# 2. as_index prevents forming of country as the second index and keeps it as a column. 
#    If you don't do this country will no longer be accesible as an column
# 3. using mean() to find the mean of each group. aggegate(np.mean()) can also be used but mean() is cythonized so is faster
mean_by_country = world_food_facts.groupby('countries', as_index = False).mean()

# define desired countries and access their means to plot
ind = mean_by_country.countries.isin(['france', 'south africa', 'united states', 'united kingdom', 'india', 'china']) 
mean_by_country.loc[ind].plot(x='countries', y='sugars_100g', kind ='bar')

# plot labelling
plt.title('Average total sugar content per 100g')
plt.ylabel('Sugar/100g')
plt.show()