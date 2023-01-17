# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud , ImageColorGenerator

%matplotlib inline
df = pd.read_csv("../input/indian-food-101/indian_food.csv")
# Print the first five rows of data

df.head()
# Print the last five rows of data

df.tail()
df.info()
#Replacing all the -1 values in the categorical and numerical columns with nan 

df= df.replace('-1',np.nan)

df = df.replace(-1,np.nan)



# Print null values

print(df.isnull().sum())
# Diet

total = len(df['diet'])

diet_values = df['diet'].value_counts()



sns.set_style("darkgrid")

sns.countplot(x="diet", data=df);

print("Vegetarian percent: ", round(diet_values[0]/total * 100),2)

print("Vegetarian percent: ", round(diet_values[1]/total * 100,2))
# Prep Time vs Cook Time

# Okey now we start explore preper time

plt.figure(figsize=(12,6))

plt.title('Prepare time')

sns.countplot(x="prep_time", data=df);
plt.figure(figsize=(12,6))

sns.lineplot(data=df, x="prep_time", y="cook_time")

plt.show()
longest_prep = df[df['prep_time'] == df['prep_time'].max()] # I used longer method, but more flexible

longest_prep
# Top 3 longest prepare dishes



df.nlargest(3,'prep_time')
plt.figure(figsize=(12,6))

plt.title("Flavor based on region")

sns.countplot(x="region",hue="flavor_profile", data=df);
plt.figure(figsize=(12,6))

plt.title("Flavor based on course")

sns.countplot(x="course",hue="flavor_profile", data=df);
def create_world_cloud(course):

    dessert_df  = df[df['course']==course].reset_index()

    ingredients = []

    for i in range(0,len(dessert_df)):

        text = dessert_df['ingredients'][i].split(',')

        text = ','.join(text)

        ingredients.append(text)

        text = ' '.join(ingredients)



    wordcloud = WordCloud(width = 400, height = 400, background_color ='white', 

                    min_font_size = 10).generate(text)                  

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis('off') 

    plt.show()
create_world_cloud("dessert")
create_world_cloud("main course")
create_world_cloud("starter")
create_world_cloud("snack")
# Create new column

df['num_ingredients'] = df['ingredients'].apply(lambda x: len(x.split(",")))
df.head()
sns.catplot(data=df, kind="bar", x="diet", y="num_ingredients", hue="flavor_profile");
df[df['num_ingredients'] == df['num_ingredients'].max()]