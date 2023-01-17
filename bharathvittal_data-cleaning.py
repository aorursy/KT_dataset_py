import numpy as np 

import pandas as pd





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/food-choices/food_coded.csv')
df.tail(10)
df.columns
cols_with_missing = [col for col in df.columns

                     if df[col].isnull().any()]
for i in cols_with_missing:

    print(i,df[i].isnull().sum())
df.shape
df.describe()
df['calories_day'].fillna(1,inplace=True)

df['comfort_food_reasons_coded'].fillna(9,inplace=True)

df['cuisine'].fillna(6,inplace=True)

df['employment'].fillna(4,inplace=True)

df['exercise'].fillna(5,inplace=True)

df['type_sports'].fillna('Nothing',inplace=True)
for i in cols_with_missing:

    df = df[~df[i].isnull()]
df.shape
df['GPA'].value_counts()
df.dropna(subset=['GPA'],inplace=True)

df['GPA_new'] = df['GPA'].str.replace(".","")

df = df[~df['GPA_new'].str.isdigit() == False]

df['GPA'] = df['GPA'].astype(float)

df.drop('GPA_new',axis=1,inplace=True)

df.head()
df.shape
df['Gender'].value_counts()
df['breakfast'].value_counts()
df['calories_chicken'].value_counts()
df['calories_day'] = df['calories_day'].astype(int)

df['calories_day'].value_counts()
df['calories_scone'] = df['calories_scone'].astype(int)

df['calories_scone'].value_counts()
df['coffee'].value_counts()
df['comfort_food'].value_counts()
df['comfort_food_reasons'].value_counts()
df['comfort_food_reasons_coded'].value_counts()
df['cook'].value_counts()
df['cuisine'].value_counts()
df['diet_current'].value_counts()
df['diet_current_coded'].value_counts()
df['drink'].value_counts()
# df['eating_changes'].value_counts()
df['eating_changes_coded'].value_counts()
df['eating_changes_coded1'].value_counts()
df['eating_out'].value_counts()
df['employment'].value_counts()
df['ethnic_food'].value_counts()
df['exercise'].value_counts()
df['father_education'].value_counts()
# df['father_profession'].value_counts()
df['fav_cuisine'].value_counts()
df['fav_cuisine_coded'].value_counts()
# df['fav_food'].value_counts()
# df['food_childhood'].value_counts()
df['fries'].value_counts()
df['fruit_day'].value_counts()
df['grade_level'].value_counts()
df['greek_food'].value_counts()
df['healthy_feeling'].value_counts()
# df['healthy_meal'].value_counts()
# df['ideal_diet'].value_counts()
df['ideal_diet_coded'].value_counts()
df['income'].value_counts()
df['indian_food'].value_counts()
df['italian_food'].value_counts()
df['life_rewarding'] = df['life_rewarding'].astype(int)

df['life_rewarding'].value_counts()
df['marital_status'] = df['marital_status'].astype(int)

df['marital_status'].value_counts()
# df['meals_dinner_friend'].value_counts()
df['mother_education'] = df['mother_education'].astype(int)

df['mother_education'].value_counts()
# df['mother_profession'].value_counts()
df['nutritional_check'].value_counts()
df['on_off_campus'] = df['on_off_campus'].astype(int)

df['on_off_campus'].value_counts()
df['parents_cook'].value_counts()
df['pay_meal_out'].value_counts()
df['persian_food'] = df['persian_food'].astype(int)

df['persian_food'].value_counts()
df['self_perception_weight'] = df['self_perception_weight'].astype(int)

df['self_perception_weight'].value_counts()
df['soup'] = df['soup'].astype(int)

df['soup'].value_counts()
df['sports'] = df['sports'].astype(int)

df['sports'].value_counts()
df['thai_food'].value_counts()
df['tortilla_calories'] = df['tortilla_calories'].astype(int)

df['tortilla_calories'].value_counts()
df['turkey_calories'].value_counts()
df['type_sports'].value_counts()
df['veggies_day'].value_counts()
df['vitamins'].value_counts()
df['waffle_calories'].value_counts()
df['weight'].value_counts()
df = df[df['weight'].str.isdigit()]
df.shape