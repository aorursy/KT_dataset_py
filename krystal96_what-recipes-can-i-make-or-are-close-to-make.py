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

          os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

recipes = pd.read_csv('/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/recipes.csv')

recipes.head(5)
def stats(x) :

    return pd.DataFrame({"Value":['%d '%x.shape[0],x.shape[1],sum(x.isnull().sum().values),

                                  sum(x.isnull().sum().values)/(recipes.shape[0]*recipes.shape[1]),]},

                        index=['Number of observations','Number of variables','Total missing value','% of Total missing value'])

summary_table=pd.DataFrame(stats(recipes))

summary_table
# Create missing value table

total = recipes.isnull().sum().sort_values(ascending=False)

percent = (recipes.isnull().sum()/recipes.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(13)
import missingno as msno

msno.bar(recipes)
plt.figure(figsize=(12, 8))

##Plot 1 Recipe Types

plt.subplot(221)

Category= pd.DataFrame({'values':recipes.groupby('Category').size().sort_values(ascending=False).values,

                        'index':recipes.groupby('Category').size().sort_values(ascending=False).index.to_list()})

order=recipes.groupby('Category').size().sort_values(ascending=False).index

sns.barplot(y='values',x='index',data=Category,order=order)

ax = plt.gca()

#Adjust xlabels for fitting the graph

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

#Add label for each cols

for a,b in zip(Category.index,Category['values']):

    plt.text(a, b+0.001, '%d' % b, ha='center', va= 'bottom',fontsize=9)

plt.title('Recipe Types')

plt.xlabel('Types')





##Plot 2 Recipe could buy by Mile

plt.subplot(222)

recipes_copy=recipes.copy()

recipes_copy.dropna(subset=['Miles Price'],how='any',inplace=True)

Category= pd.DataFrame({'values':recipes_copy.groupby('Category').size().sort_values(ascending=False).values,

                        'index':recipes_copy.groupby('Category').size().sort_values(ascending=False).index.to_list()})

order=recipes_copy.groupby('Category').size().sort_values(ascending=False).index

sns.barplot(y='values',x='index',data=Category,order=order)

ax = plt.gca()

#Add label for each cols

for a,b in zip(Category.index,Category['values']):

    plt.text(a, b+0.001, '%d' % b, ha='center', va= 'bottom',fontsize=9)

#Add label for each cols

plt.xlabel('Types')

plt.title('Recipe could buy by Mile')





##Plot 3 Recipe Source

plt.subplot(212)

Source= pd.DataFrame({'values':recipes.groupby('Source').size().sort_values(ascending=False).values,

                        'index':recipes.groupby('Source').size().sort_values(ascending=False).index.to_list()})

order_Source=recipes.groupby('Source').size().sort_values(ascending=False).index

sns.barplot(y='values',x='index',data=Source,order=order_Source)

ax = plt.gca()

#Adjust xlabels for fitting the graph

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

#Add label for each cols

for a,b in zip(Source.index,Source['values']):

    plt.text(a, b+0.001, '%d' % b, ha='center', va= 'bottom',fontsize=9)

plt.title('Recipe Source')

plt.xlabel('Source')



plt.show()
def close_to_make(what_in_my_bag):

    close_to_make=[]

    df=recipes.iloc[:,:13]

    for i in range(len(df)):

        for j in range(df.shape[1]):   

            if str(df.iloc[i,j]) in what_in_my_bag.keys()  :

                close_to_make.append(str(df.iloc[i,0])) 

                break

    return pd.DataFrame({"What is close to make":close_to_make })

what_in_my_bag={'apple':5 }

close_to_make(what_in_my_bag)
def could_make(what_in_my_bag):

    could_make = []

    df=recipes.iloc[:,:13]

    for i in range(len(df)):

        recipe = {}

        for j in (2, 4, 6, 8, 10, 12):

            if str(df.iloc[i,j]) != 'nan':

                recipe[str(df.iloc[i,j])]=df.iloc[i,j-1]



        count = len(recipe)

        for k in recipe.keys():

            if k in what_in_my_bag.keys():

                if recipe[k] <= what_in_my_bag[k]:

                    count -= 1

            else:

                break

        if count == 0:

            could_make.append(str(df.iloc[i,0]))

    return pd.DataFrame({"What could I make":could_make})

what_in_my_bag={'apple':10,'wood':4 }

could_make(what_in_my_bag)
