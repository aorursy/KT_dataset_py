import numpy as np

import pandas as pd
indian_recipes = "../input/indian-food-101/indian_food.csv"

df_indian_recipes = pd.read_csv(indian_recipes)

print("shape", df_indian_recipes.shape, sep=": ")



print("column types",df_indian_recipes.dtypes, sep=":\n")
df_indian_recipes.replace(-1, np.NaN, inplace = True)

df_indian_recipes.replace("-1", np.NaN, inplace = True)

df_indian_recipes.nunique()
df_indian_recipes.head()
ingredients = set()

for item in df_indian_recipes['ingredients']:

    ingredients.update(str(item).lower().split(","))

    

print("Total unique ingredients in dataset",len(ingredients),sep=": ")
print("Are there any NA values in any column", df_indian_recipes.isna().sum(), sep=":\n")
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (9, 5)

matplotlib.rcParams['figure.facecolor'] = '#00000000'
recipe_by_state = df_indian_recipes.groupby('state').size().to_frame(name = "count").reset_index()

sns.barplot(x = 'count', y='state', data = recipe_by_state )



plt.title("Recipes by State")

plt.xlabel("State")

plt.ylabel("Count of recipes")



plt.show()
df_diet_type = df_indian_recipes.diet.value_counts().reset_index()

plt.pie(df_diet_type.diet, labels = df_diet_type['index'],autopct='%1.1f%%')

plt.title("Vegetarian vs Non-Vegetarian recipes in dataset")

plt.show()
df_course = df_indian_recipes.course.value_counts().reset_index()

sns.barplot(x = 'course', y = 'index', data = df_course)



plt.title("Cuisines")

plt.show()
df_cook_time = (df_indian_recipes.prep_time + df_indian_recipes.cook_time).to_frame('total_time').reset_index()

plt.hist(df_cook_time['total_time'],np.arange(5,150,10))



plt.title("Cooking time")

plt.ylabel("Number of recipes")

plt.xlabel("Time in minutes")



plt.show()
df_temp = df_indian_recipes

df_temp['total_time'] = df_indian_recipes.prep_time + df_indian_recipes.cook_time

df_temp.sort_values('total_time',ascending = False).head()[['name','course','total_time']]
df_up_dishes = (df_indian_recipes[df_indian_recipes['state'] == "Uttar Pradesh"][['name','ingredients']])

def count_ingredient(column):

    return len(column.split(","))

df_up_dishes['ingredient_count'] = df_up_dishes['ingredients'].apply(count_ingredient)
df_up_dishes.sort_values('ingredient_count', ascending = False).head()
df_temp[(df_temp['diet'] == 'vegetarian') & (df_temp['course'] == 'main course')].sort_values("total_time", ascending= False).head()
df_temp[(df_temp['diet'] == 'non vegetarian') & (df_temp['course'] == 'main course')].sort_values('total_time', ascending = False).head()
df_temp[(df_temp['course'] == 'dessert')].sort_values('total_time', ascending = False).head()