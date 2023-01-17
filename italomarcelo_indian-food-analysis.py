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
import plotly.express as px
df = pd.read_csv('/kaggle/input/indian-food-101/indian_food.csv')
# function Exploratory Data Analysis

def eda(dfA, all=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nDTypes - Numerics')

    print(dfA.select_dtypes(include=np.number).columns.tolist())

    print(f'\nDTypes - Categoricals')

    print(dfA.select_dtypes(include='object').columns.tolist())

    print(f'\nIs Null: {dfA.isnull().sum().sum()}')

    print(f'{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if all:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))



# function Fill NaN values

def cleanNaN(dfA):

  for col in dfA:

    if type(dfA[col]) == 'object':

        dfA[col] = dfA[col].fillna('unknow')

    else:

        dfA[col] = dfA[col].fillna(0)

  return dfA
eda(df)
df.region.unique() # only columns with null value
cleanNaN(df)

eda(df)
pd.set_option('display.max_colwidth', None)

df.sample(2)
ingredientsAll = []

for k in df.ingredients.values.tolist():

    for i in k.split(','):

        ingredientsAll.append(i.strip())
ingredients = pd.value_counts(ingredientsAll)

ingredients
ing20=ingredients[:20]

ing20
fig = px.bar(ing20, color=ing20.index, title='Top 20 - Ingredients')

fig.show()
diets = df.diet.unique()

diets
xd = df.diet.value_counts()

fig = px.pie(xd, values=xd.values, names=xd.index, title='Diets', 

             color=xd.index, color_discrete_sequence=px.colors.sequential.Greens_r)

fig.show()
flavors = df['flavor_profile'].unique()

flavors
df.loc[df['flavor_profile']=='-1','flavor_profile'] = 'unknow'
flavors = df['flavor_profile'].unique()

flavors
xd = df['flavor_profile'].value_counts()

fig = px.pie(xd, values=xd.values, names=xd.index, title='Flavors', 

             color=xd.index, color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
courses = df.course.unique()

courses
xd = df.course.value_counts()

fig = px.pie(xd, values=xd.values, names=xd.index, title='Courses', 

             color=xd.index, color_discrete_sequence=px.colors.sequential.YlGnBu)

fig.show()
states = df.state.unique()

states
st = df.state.value_counts()

fig = px.bar(st, color=st.index, title='States')

fig.show()
regions = df.region.unique()

regions
df.loc[df.region=='-1','region'] = 'unknow'

df.loc[df.region==0,'region'] = 'unknow'

regions = df.region.unique()

regions
xd = df.region.value_counts()

fig = px.pie(xd, values=xd.values, names=xd.index, title='Regions', 

             color=xd.index, color_discrete_sequence=px.colors.sequential.Electric_r)

fig.show()
df.prep_time.unique()
# I'll considere prep_time with mean to values equal -1

pt = int(df.prep_time.mean())

df.loc[df.prep_time == -1,'prep_time'] = pt
xd = df.prep_time.value_counts()[:5]

xd
fig = px.bar(x=xd.index, y=xd.values, color=xd.index, title='Preparation Time (min)',

            labels=dict(x='minutes', y='qty of plates'))

fig.show()
df.cook_time.unique()
# I'll considere cook_time with mean to values equal -1

ct = int(df.cook_time.mean())

df.loc[df.cook_time == -1,'cook_time'] = ct
xd = df.cook_time.value_counts()[:5]

xd
fig = px.bar(x=xd.index, y=xd.values, color=xd.index, title='Cooking Time (min)',

            labels=dict(x='minutes', y='qty of plates'))

fig.show()
# creating ingredients list

listIngredients = ingredients.index.tolist()
# return % similarity betweens 2 lists

def similarityArrays(t1,t2):

    return len(set(t1) & set(t2)) / float(len(set(t1) | set(t2))) * 100
# convert ingredients to numeric array

def convertIngredients(listIng):

    li = []

    for ing in listIng.split(','):

        ing = ing.strip()

        li.append(listIngredients.index(ing))

    return li
# get one sample 

ingSample = df.ingredients.head(1).values[0]

ingSample
# double check in function

for teste in convertIngredients(ingSample):

    print(listIngredients[teste], end=' ')
# testing function in lambda

df.ingredients.head(1).apply(lambda x: convertIngredients(x))
df['ingredientsList'] = df.ingredients.apply(lambda x: convertIngredients(x))
# look at sugar and ghee

df[['ingredients','ingredientsList']].head()
dfs = pd.DataFrame()

for a in range(0, len(df)):

    dishA = df.name.iloc[a]

    dishAlist = df.ingredientsList.iloc[a]

    for b in range(0, len(df)):

        if a != b:

            dishB = df.name.iloc[b]

            dishBlist = df.ingredientsList.iloc[b]

            s = similarityArrays(dishAlist, dishBlist)

            dfs = dfs.append({'plate A': dishA, 'plate B': dishB, 'similarity': s}, 

                             ignore_index=True)

dfs[dfs.similarity >50].sort_values(by='similarity', ascending=False)
df[df.name=='Pattor']
df[df.name=='Patra']