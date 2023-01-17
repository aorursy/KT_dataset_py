import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from wordcloud import WordCloud



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%config InlineBackend.figure_format = 'retina'

plt.rc('font', family = 'AppleGothic')
df = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')

df.head()
# -1 == NaN

df = df.replace(['-1', -1], np.nan)



## check null data

msno.matrix(df)

plt.show()
df.isnull().sum()
df_diet = df['diet'].value_counts().reset_index()



plt.rcParams['font.size'] = 12

plt.rcParams['font.weight'] = 'bold'



fig, ax = plt.subplots()

ax.pie(data=df_diet,

       x='diet',

       labels='index',

       autopct='%1.1f%%',

       explode=[0, 0.2])

ax.set_title('Ratio of diet', fontsize=15, fontweight='bold')

plt.show()
fig, axes = plt.subplots(figsize=(10, 4))

sns.countplot(data=df, x='region', hue='diet', ax=axes)

axes.set_title('Relationship between region and vegetarian',

               fontsize=15,

               fontweight='bold')

axes.legend(loc='upper center', bbox_to_anchor=(1.15, 1))

plt.show()
df_course = df['course'].value_counts().reset_index()



fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(

    data=df_course,

    x='course',

    labels='index',

    autopct='%1.1f%%',

)



sns.countplot(data = df, x = 'course', hue = 'diet', order=df_course['index'], ax = axes[1])

plt.suptitle('Ratio of course', fontsize=20, fontweight='bold', x= 0.5, y = 1.1)

plt.tight_layout()

plt.show()
df['Nb_of_ingredients'] = df['ingredients'].map(lambda x: len(x.split(',')))

df.head()
fig, axes = plt.subplots(figsize=(8, 4))

sns.barplot(data=df, x='course', y='Nb_of_ingredients', ax=axes)

plt.title('Number of ingredients for Course', fontweight='bold')

plt.show()
def concat_ingredients(df):

    ingreds = df['ingredients'].map(lambda x: x.split(','))



    text = []

    for ingred in ingreds:

        ingred = ','.join(ingred)

        text.append(ingred)

    text = ', '.join(text)

    return text
courses = df['course'].dropna().unique()



texts = []



for course in courses:

    df_course = df[df['course'] == course]

    texts.append(concat_ingredients(df_course))



fig, axes = plt.subplots(1, 4, figsize=(20, 5))



idx = 0



for ax, text in zip(axes, texts):

    wordcloud = WordCloud(width=400,

                          height=400,

                          background_color='white',

                          min_font_size=10).generate(text)

    ax.imshow(wordcloud, interpolation='hanning')

    ax.set_title(courses[idx], fontweight='bold', fontsize=20)

    ax.axis('off')



    idx += 1



plt.show()
df_time = df[['prep_time', 'cook_time', 'course']]

df_time_melt = pd.melt(df_time, id_vars='course', value_vars=['prep_time', 'cook_time'], var_name='kind', value_name='time')



fig, axes = plt.subplots(figsize = (10, 6))

sns.barplot(data = df_time_melt, x = 'course', y='time', hue = 'kind', ax = axes)

axes.set_title('Preparing and cooking time', fontweight = 'bold', fontsize = 20)

plt.show()
fig, axes = plt.subplots(figsize=(10, 6))

sns.countplot(data=df, x='course', hue='flavor_profile', ax=axes)

axes.legend(loc='upper center', bbox_to_anchor=(1.15, 1))

axes.set_title('Flavor profile about course', fontweight='bold', fontsize=20)

plt.show()
fig, axes = plt.subplots(figsize=(10, 6))

sns.countplot(

    data=df,

    x='region',

    hue='flavor_profile',

)

plt.legend(bbox_to_anchor=(1, 1))

plt.show()
regions = df['region'].dropna().unique()



texts = []

for region in regions:

    df_reg = df[df['region'] == region]

    texts.append(concat_ingredients(df_reg))



fig, axes = plt.subplots(1, 6, figsize=(30, 5))



idx = 0

for ax, text in zip(axes, texts):

    wordcloud = WordCloud(width=400,

                          height=400,

                          background_color='white',

                          min_font_size=10).generate(text)

    ax.imshow(wordcloud, interpolation = 'hanning')

    ax.set_title(regions[idx], fontweight='bold', fontsize=15)

    ax.axis('off')

    

    

    idx += 1