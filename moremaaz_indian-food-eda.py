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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



sns.set(rc= {'figure.figsize': (12,8)})

plt.style.use('ggplot')



from plotly.offline import init_notebook_mode, iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly.graph_objs as go

import plotly

import plotly.express as px

import plotly.figure_factory as ff
df = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
df.info()
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset')
df.head()
df.isna().sum()
df.region.dropna(inplace=True)
df.describe()
df = df.replace('-1', np.nan)

df = df.replace(-1, np.nan)
df.select_dtypes([np.int, np.float]).nunique().iplot(kind='bar', 

                                                     labels= 'index', values= 'values', title="Unique Count: Numeric Columns", color= 'blue')
px.histogram(df.prep_time, title= 'Distribution of Prep Time')
px.histogram(df.cook_time, title= 'Distribution of Cook Time')
df.select_dtypes(object).nunique().sort_values(ascending= False).iplot(kind='bar', 

                                                     labels= 'index', values= 'values', title="Unique Count: Discrete Variables", color= 'orange')
pie = df.region.value_counts()

pie_df = pd.DataFrame({'index':pie.index, 'values': pie.values})

pie_df.iplot(kind='pie', labels= 'index', values= 'values', hole= .5, title="Value counts: region")



pie2 = df.course.value_counts()

pie_df2 = pd.DataFrame({'index':pie2.index, 'values': pie2.values})

pie_df2.iplot(kind='pie', labels= 'index', values= 'values', hole= .5, title="Value counts: course")



pie3 = df.flavor_profile.value_counts()

pie_df3 = pd.DataFrame({'index':pie3.index, 'values': pie3.values})

pie_df3.iplot(kind='pie', labels= 'index', values= 'values', hole= .5, title="Value counts: flavor profile")



pie4 = df.diet.value_counts()

pie_df4 = pd.DataFrame({'index':pie4.index, 'values': pie4.values})

pie_df4.iplot(kind='pie', labels= 'index', values= 'values', hole= .5, title="Value counts: diet")
bar1 = df.state.value_counts().reset_index()

bar1.columns = ['state', 'count']

px.bar(bar1, x='state', y='count', title= 'State Count', color_discrete_sequence= px.colors.qualitative.Prism, 

labels={

        'state': 'State',

         'count': 'Count'

        })
bar2 = df.region.value_counts().reset_index()

bar2.columns = ['region', 'count']

px.bar(bar2, x='region', y='count', title= 'Region Count', color_discrete_sequence= px.colors.qualitative.Dark2, 

      labels={

                     'region': 'Region',

                     'count': 'Count'

                 })
bar3 = df.course.value_counts().reset_index()

bar3.columns = ['course', 'count']

px.bar(bar3, x='course', y='count', title= 'Course Count', color_discrete_sequence= px.colors.qualitative.Vivid,

labels={

         'course': 'Course',

         'count': 'Count'

        })
bar4 = df.flavor_profile.value_counts().reset_index()

bar4.columns = ['flavor', 'count']

px.bar(bar4, x='flavor', y='count', title= 'Flavor Count', color_discrete_sequence= px.colors.qualitative.Set3, 

labels={

         'flavor': 'Flavor',

         'count': 'Count'

        })
top_10_prep = df.loc[df.prep_time >= 1, ['name', 'prep_time']].sort_values(by='prep_time', ascending=False).head(10)

px.bar(top_10_prep, y='name', x='prep_time', color='prep_time', title= 'Top Ten Longest Prep Time Dishes', 

labels={

        'name': 'Dish',

         'prep_time': 'Prep Time (min)'

        })
top_10_cook = df.loc[df.cook_time >= 1, ['name', 'cook_time']].sort_values(by='cook_time', ascending=False).head(10)

px.bar(top_10_cook, y='name', x='cook_time', color='cook_time', title= 'Top Ten Longest Cook Time Dishes', 

labels={

        'name': 'Dish',

         'cook_time': 'Cook Time (min)'

        })
df.groupby('diet')[['prep_time', 'cook_time']].mean().iplot(kind= 'bar', title= 'Mean Cooking vs Prep Time for Diet Type')
px.scatter(df, x='prep_time', y='cook_time', color= 'diet', title= 'Scatterplot: Cook time vs Prep time',

                    labels={

                     'prep_time': 'Prep Time (min)',

                     'cook_time': 'Cook Time (min)',

                        'diet' : 'Diet'

                 })
d = df.groupby(['region', 'diet']).mean().reset_index()

px.bar(d, x='region', y='prep_time', color='diet', color_discrete_sequence= 

       px.colors.qualitative.Dark2, title='Average Prep Time for Region and Diet', 

      labels={ 'region': 'Region',

              'prep_time': 'Prep Time (min)',

              'diet' : 'Diet'})
px.bar(d, x='region', y='cook_time', color='diet', color_discrete_sequence= 

       px.colors.qualitative.Antique, title='Cook Time for Region and Diet', 

            labels={ 'region': 'Region',

              'cook_time': 'Cook Time (min)',

              'diet' : 'Diet'})
pd.options.plotting.backend = "plotly"

df.groupby('course').sum().plot.area(title='Sum of Cook vs Prep Time per Course', color_discrete_sequence= 

       px.colors.qualitative.Safe, 

                        labels={'course': 'Course',

                        'value': 'Minutes',

                        'variable' : 'Legend'})
df.groupby(['course', 'flavor_profile']).mean().iplot(kind='line', title= 'Mean Prep vs Cook Time Per Flavour Profile & Course')
ingredients = pd.Series(df.ingredients.str.split(',').sum()).value_counts()

ingredients= ingredients[ingredients>12]



px.bar(ingredients, y=ingredients.values, x=ingredients.index, color=ingredients.values, title= 'Top Ten Ingredients in Dishes', 

    labels={

    'index': 'Igredient',

    'y': 'Count'

        })
px.parallel_categories(df.drop(['state', 'region', 'cook_time'], axis=1), 

                       color="prep_time", color_continuous_scale=px.colors.sequential.Agsunset,

                      title='Parallel Categories')
def get_text(column):

    words = ''

    for text in column:

        words += text

    return words
veg = df.loc[df.diet == 'vegetarian', 'ingredients']



text1 = get_text(veg)



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text1)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()
nonveg = df.loc[df.diet == 'non vegetarian', 'ingredients']



text2 = get_text(nonveg)



stopwords = set(STOPWORDS)

wc = WordCloud(background_color= 'black', stopwords= stopwords,

              width=1600, height=800)



wc.generate(text2)

plt.figure(figsize=(20,10), facecolor='k')

plt.axis('off')

plt.tight_layout(pad=0)

plt.imshow(wc)

plt.show()