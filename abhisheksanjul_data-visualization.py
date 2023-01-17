# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

wff_data = pd.read_csv('../input/FoodFacts.csv',encoding='UTF-8')

wff_data = wff_data[['product_name','countries_en','vitamin_a_100g','energy_100g','carbohydrates_100g','main_category_en','fat_100g','nutrition_score_fr_100g','nutrition_score_uk_100g','sugars_100g','fiber_100g','proteins_100g']]

wff_data = wff_data[wff_data.countries_en.notnull() & wff_data.main_category_en.notnull()]

#print(len(wff_data.main_category_en.unique()))

# checking the relation between carbohydrates_100g and energy_100g 



def energy_vs_carbo(country):

 energy = wff_data.energy_100g[wff_data.countries_en == country]

 carbo = wff_data.carbohydrates_100g[wff_data.countries_en == country]

 plt.scatter(energy,carbo)

 plt.ylabel("carbohydrates_100g")

 plt.xlabel("energy_100g")

 plt.show()

 

energy_vs_carbo('United States') 

energy_vs_carbo('France') 



def energy_vs_fat(country):

 energy = wff_data.energy_100g[wff_data.countries_en == country]

 fat = wff_data.fat_100g[wff_data.countries_en == country]

 plt.scatter(energy,fat)

 plt.ylabel("fat_100g")

 plt.xlabel("energy_100g")

 plt.show()

 

energy_vs_fat('United States') 

energy_vs_fat('France')



def energy_vs_sugar(country):

 energy = wff_data.energy_100g[wff_data.countries_en == country]

 sugars = wff_data.sugars_100g[wff_data.countries_en == country]

 plt.scatter(energy,sugars)

 plt.ylabel("sugars_100g")

 plt.xlabel("energy_100g")

 plt.show()

 

energy_vs_sugar('United States') 

energy_vs_sugar('France')



def energy_vs_fiber(country):

 energy = wff_data.energy_100g[wff_data.countries_en == country]

 fiber = wff_data.fiber_100g[wff_data.countries_en == country]

 plt.scatter(energy,fiber)

 plt.ylabel("fiber_100g")

 plt.xlabel("energy_100g")

 plt.show()

energy_vs_fiber('United States') 

energy_vs_fiber('France')



def energy_vs_proteins(country):

 energy = wff_data.energy_100g[wff_data.countries_en == country]

 proteins = wff_data.proteins_100g[wff_data.countries_en == country]

 plt.scatter(energy,proteins)

 plt.ylabel("proteins_100g")

 plt.xlabel("energy_100g")

 plt.show()

 

energy_vs_proteins('United States') 

energy_vs_proteins('France')



# France nutrition with energy relation

plt.scatter(wff_data.energy_100g,wff_data.nutrition_score_fr_100g)

plt.ylabel('nutrition_score_fr_100g')

plt.xlabel('energy_100g')

plt.show()



#UK nutrition with energy 

plt.scatter(wff_data.energy_100g,wff_data.nutrition_score_uk_100g)

plt.ylabel('nutrition_score_uk_100g')

plt.xlabel('energy_100g')

plt.show()
