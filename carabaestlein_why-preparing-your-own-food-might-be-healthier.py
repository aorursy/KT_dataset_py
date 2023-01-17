import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import linear_model

from sklearn.metrics import mean_squared_error,  r2_score



off_df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', sep='\t', low_memory=False)
off_df['code'].count()
off_df['ingredients_text'].count()
off_df['ingredients_text'].dtype
off_df['number_of_ingredients']= off_df['ingredients_text'].str.split(',').str.len()

off_df=off_df.dropna(subset=['number_of_ingredients']) 

off_df['number_of_ingredients']=off_df['number_of_ingredients'].astype(int)

off_df.index = range(len(off_df))

off_df[['ingredients_text', 'number_of_ingredients']][:25]
off_df['number_of_ingredients'].hist(bins=100).set_xlabel('Number of Ingredients')
off_df['number_of_ingredients'].quantile(q=0.9)
off_df['number_of_ingredients'].max()
off_df['number_of_ingredients'].median()
off_df['number_of_ingredients'].mean()
off_df['number_of_ingredients'].hist(bins=100).set_xlim((0,20))
off_df['number_of_ingredients_categories']='many ingredients'

off_df['number_of_ingredients_categories']=np.where(off_df['number_of_ingredients']>5, off_df['number_of_ingredients_categories'],'5 or less ingredients')

off_df['number_of_ingredients_categories']=np.where(off_df['number_of_ingredients']!=1, off_df['number_of_ingredients_categories'],'1 ingredient')

off_df['number_of_ingredients_categories'].value_counts()
off_df[['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']].groupby(off_df['number_of_ingredients_categories']).mean()
off_df[['fat_100g', 'saturated-fat_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'carbohydrates_100g']].groupby(off_df['number_of_ingredients_categories']).mean().plot().set_xlabel('Number of Ingredients')
off_df[['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']].groupby(off_df['number_of_ingredients_categories']).median()
off_df[['fat_100g', 'saturated-fat_100g', 'sugars_100g', 'proteins_100g', 'salt_100g', 'carbohydrates_100g']].groupby(off_df['number_of_ingredients_categories']).median().plot().set_xlabel('Number of Ingredients')
off_df[['fat_100g', 'saturated-fat_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']].groupby(off_df['number_of_ingredients_categories']).median().plot().set_xlabel('Number of Ingredients')
off_df['pnns_groups_2'].value_counts()
off_df['product_types']='unknown'



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Non-sugared beverages', off_df['product_types'],'beverages')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Sweetened beverages', off_df['product_types'],'beverages')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Alcoholic beverages', off_df['product_types'],'beverages')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Fruit juices', off_df['product_types'],'beverages')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Artificially sweetened beverages', off_df['product_types'],'beverages')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='One-dish meals', off_df['product_types'],'one_dish_meals')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Sweets', off_df['product_types'],'sweets')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Biscuits and cakes', off_df['product_types'],'sweets')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Chocolate products', off_df['product_types'],'sweets')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Ice cream', off_df['product_types'],'sweets')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Dairy desserts', off_df['product_types'],'sweets')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Cereals', off_df['product_types'],'cereals')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Breakfast cereals', off_df['product_types'],'cereals')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='cereals', off_df['product_types'],'cereals')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Cheese', off_df['product_types'],'dairy')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Milk and yogurt', off_df['product_types'],'dairy')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Processed meat', off_df['product_types'],'meat_and_fish')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Fish and seafood', off_df['product_types'],'meat_and_fish')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Meat', off_df['product_types'],'meat_and_fish')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Eggs', off_df['product_types'],'eggs')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Vegetables', off_df['product_types'],'vegetables')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='vegetables', off_df['product_types'],'vegetables')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Legumes', off_df['product_types'],'vegetables')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='legumes', off_df['product_types'],'vegetables')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Potatoes', off_df['product_types'],'vegetables')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Fruits', off_df['product_types'],'fruit')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Dried fruits', off_df['product_types'],'fruit')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Fruit nectars', off_df['product_types'],'fruit')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='fruits', off_df['product_types'],'fruit')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Nuts', off_df['product_types'],'nuts')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='nuts', off_df['product_types'],'nuts')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Fats', off_df['product_types'],'fats')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Bread', off_df['product_types'],'bread')



off_df['product_types']=np.where(off_df['pnns_groups_2']!='Dressings and sauces', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Appetizers', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Sandwich', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Pizza pies and quiche', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='pastries', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Soups', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Tripe dishes', off_df['product_types'],'ready_made_foods')

off_df['product_types']=np.where(off_df['pnns_groups_2']!='Salty and fatty products', off_df['product_types'],'ready_made_foods')
off_df['number_of_ingredients_categories'].groupby([off_df['product_types']]).value_counts()
off_df[['number_of_ingredients']].groupby(off_df['product_types']).median()
off_df[['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']].groupby([ off_df['product_types'], off_df['number_of_ingredients_categories']]).mean()
off_df[['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']].groupby([ off_df['product_types'], off_df['number_of_ingredients_categories']]).median()
regr = linear_model.LinearRegression()

column_heads = ['carbohydrates_100g', 'sugars_100g', 'fat_100g', 'saturated-fat_100g', 'proteins_100g', 'salt_100g']

off_df['constant']= 1

results = list()

for i in column_heads:

    off_df_temp = off_df.dropna(subset=[['number_of_ingredients', i]])

    off_df_temp = off_df_temp[off_df_temp['number_of_ingredients']<50]

    off_df_temp_train = off_df_temp[:-200]

    off_df_temp_test = off_df_temp[-200:]

    regr.fit(off_df_temp_train[['constant', 'number_of_ingredients']], off_df_temp_train[i])

    off_df_pred = regr.predict(off_df_temp_test[['constant', 'number_of_ingredients']])

    error = mean_squared_error(off_df_temp_test[i], off_df_pred)

    R2 = r2_score(off_df_temp_test[i], off_df_pred)

    results.append([i, regr.coef_[1], error, R2])

    plt.figure()

    plt.scatter(off_df_temp_test[['number_of_ingredients']], off_df_temp_test[i],  color='black')

    plt.scatter(off_df_temp_test[['number_of_ingredients']], off_df_pred, color='blue')

    plt.suptitle(i)

results
off_df['product_types'].value_counts()
product_types = ['sweets', 'beverages', 'ready_made_foods', 'meat_and_fish', 'dairy', 'one_dish_meals', 'cereals', 'vegetables', 'fruit', 'fats', 'bread', 'nuts', 'eggs'] 

for i in product_types:

    off_df[i] = np.where(off_df['product_types']==i, 1,0)

results_control = list()

for i in column_heads:

    off_df_temp = off_df.dropna(subset=[['number_of_ingredients', i]])

    off_df_temp = off_df_temp[off_df_temp['number_of_ingredients']<50]

    off_df_temp_train = off_df_temp[:-200]

    off_df_temp_test = off_df_temp[-200:]

    regr.fit(off_df_temp_train[['constant', 'number_of_ingredients','sweets', 'beverages', 'ready_made_foods', 'meat_and_fish', 'dairy', 'one_dish_meals', 'cereals', 'vegetables', 'fruit', 'fats', 'bread', 'nuts', 'eggs']], off_df_temp_train[i])

    off_df_pred = regr.predict(off_df_temp_test[['constant', 'number_of_ingredients', 'sweets', 'beverages', 'ready_made_foods', 'meat_and_fish', 'dairy', 'one_dish_meals', 'cereals', 'vegetables', 'fruit', 'fats', 'bread', 'nuts', 'eggs']])

    error = mean_squared_error(off_df_temp_test[i], off_df_pred)

    R2 = r2_score(off_df_temp_test[i], off_df_pred)

    results_control.append([i, regr.coef_[1], error, R2])

    plt.figure()

    plt.scatter(off_df_temp_test[['number_of_ingredients']], off_df_temp_test[i],  color='black')

    plt.scatter(off_df_temp_test[['number_of_ingredients']], off_df_pred, color='blue')

    plt.suptitle(i)

results_control