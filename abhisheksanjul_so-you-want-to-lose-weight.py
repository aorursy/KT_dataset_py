# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





food_doc = pd.read_csv('../input/FoodFacts.csv')

food_doc_specs = food_doc[food_doc.energy_100g.notnull() & food_doc.carbohydrates_100g.notnull() & food_doc.fat_100g.notnull() & food_doc.proteins_100g.notnull() & food_doc.main_category_en.notnull() & food_doc['nutrition_score_uk_100g']]



food_doc_specs = food_doc_specs[['code','generic_name','energy_100g','carbohydrates_100g','fat_100g','proteins_100g', 'main_category_en', 'nutrition_score_uk_100g']]





food_doc_counts = food_doc_specs.groupby('main_category_en').filter(lambda x:len(x)>60)



list = ['Sugary snacks', 'Dairies', 'Meats', 'Beverages', 'Salty snacks', 'Fruit juices', 'Canned foods', 'Desserts', 'Seafood', 'Spreads', 'Sweeteners', 'es:Pan-y-reposteria']

print (food_doc_counts[food_doc_counts['main_category_en' != list]])



fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = [15,8], sharex = True)

plt.xlim((0,5000))

plot_carbs = sns.regplot(x = "energy_100g", y = "carbohydrates_100g", fit_reg = False, ax = ax1, data = food_doc_specs)

plot_fat = sns.regplot(x = "energy_100g", y = "fat_100g", ax = ax2, fit_reg = False, data = food_doc_specs)

plot_protein = sns.regplot(x = "energy_100g", y = "proteins_100g", fit_reg = False, ax = ax3, data = food_doc_specs)
g_carbs = sns.FacetGrid(food_doc_counts, col = "main_category_en", col_wrap = 5, ylim = (0,160), xlim = (0,2000))

g_carbs = g_carbs.map(plt.scatter,"energy_100g", "carbohydrates_100g")
g_fats = sns.FacetGrid(food_doc_counts, col = "main_category_en", col_wrap = 5, ylim = (0,60), xlim = (0,2000))

g_fats = g_fats.map(plt.scatter, "energy_100g", "fat_100g", color = "green")
g_proteins = sns.FacetGrid(food_doc_counts, col = "main_category_en", col_wrap = 5, ylim = (0,60), xlim = (0,2000))

g_proteins = g_proteins.map(plt.scatter, "energy_100g","proteins_100g", color = "red")
mean_nutrients = food_doc_counts.groupby('main_category_en', as_index = False).mean()





g_mean_calories = sns.FacetGrid(mean_nutrients,col = "main_category_en", col_wrap = 5)

g_mean_calories =g_mean_calories.map(sns.barplot, "energy_100g")
new_means = pd.melt(mean_nutrients, id_vars = ['main_category_en'], value_vars = ['carbohydrates_100g', 'proteins_100g','fat_100g'])



g_new_means = sns.factorplot(x = 'variable', y = 'value', col = 'main_category_en', kind = 'bar', col_wrap = 5, legend = True, data = new_means)



g_nutrition_grades = sns.FacetGrid(food_doc_counts, col = "main_category_en", col_wrap = 5)

g_nutrition_grades = g_nutrition_grades.map(sns.boxplot, "nutrition_score_uk_100g")