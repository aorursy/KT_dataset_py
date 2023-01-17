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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

%matplotlib inline

sns.set_style('darkgrid')
data = pd.read_csv('../input/indian-food-101/indian_food.csv')
data.head()
data.isna().sum()
# Null values are indicated by -1 so null values will be converted to -1 for uniformity



data.loc[data['region'].isna(),'region']='-1'
for col in data.columns:

    print(col,' : ',len(data[data[col]==-1].index))
data.describe()
data.nunique()
# Percentage of NULL Values. Here, -1 indicates null values and '-1' shows null values in region column.



for col in data.columns[:-1]:

    print(col,' : ',round(len(data[data[col]==-1].index)/len(data)*100,2))

print('region : ',round(len(data[data['region']=='-1'].index)/len(data)*100,2))
# We cant add the 'prep_time' and 'cook_time' columns to chaeck the total time as 

# there are different amount of null values in both columns.
plt.figure(figsize=(8,6))

sns.countplot(data['diet'])

plt.title('DIET Countplot')

plt.show()
data['flavor_profile'].value_counts().plot(kind='bar',figsize=(8,6))

plt.title('FLAVOR PROFILE Countplot')

plt.show()
data['course'].value_counts().plot(kind='bar',figsize=(8,6))

plt.title('COURSE Countplot')

plt.show()
data['region'].value_counts().plot(kind='bar',figsize=(12,6))

plt.title('REGION-WISE Countplot')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(2,2,1)

plt.pie(data['diet'].value_counts(),autopct='%1.2f%%',labels=data['diet'].unique())

plt.title('Diet Type Distribution Percentage')

plt.subplot(2,2,2)

plt.pie(data['flavor_profile'].value_counts(),autopct='%1.2f%%',labels=data['flavor_profile'].unique())

plt.title('Flavor Profile Distribution Percentage')

plt.subplot(2,2,3)

plt.pie(data['course'].value_counts(),autopct='%1.2f%%',labels=data['course'].unique())

plt.title('Course Distribution Percentage')

plt.subplot(2,2,4)

plt.pie(data['region'].value_counts(),autopct='%1.2f%%',labels=data['region'].unique())

plt.title('Region Distribution Percentage')

plt.show()
data['state'].value_counts().plot(kind='bar',figsize=(12,6))

plt.title('STATE-WISE Countplot')

plt.show()
plt.figure(figsize=(15,15))

plt.pie(data['state'].value_counts(),autopct='%1.2f%%',labels=data['state'].unique())

plt.title('State Percentage')

plt.show()
data['num_ingd'] = data.ingredients.apply(lambda x : len(x.split(', ')))
plt.figure(figsize=(12,6))

sns.distplot(data['prep_time'])

plt.title('PREPARATION TIME Distribution')

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(data['cook_time'])

plt.title('COOKING TIME Distribution')

plt.show()
plt.figure(figsize=(8,4))

sns.distplot(data.num_ingd)

plt.xticks([i for i in range(0,12)])

plt.title('NUMBER OF INGREDIENTS Distribution')

plt.show()
data.head()
ingredients = [ingd.split(', ') for ingd in data.ingredients] # List form of Ingredients for each food item
# Strip is necessary as there are lot of elements which are similar but contains extra spaces as prefix or suffix like

# ' jaggery', 'jaggery' and 'jaggery ' and lower is used because the first character of every ingrediients list is uppercase but

# if the same element is not at first index then it will be considered as seperate element.

# There are some elements which have the same meaning but but are in different languages like jaggery and gur.

# This can be resolved by language translational techniques and it will become very messy.



all_ingd = list(set([ing.lower().strip() for ingd in ingredients for ing in ingd])) # Set of all the ingredients

all_ingd.sort()

len(all_ingd)
# Vectorising the ingredients



ingd_vec = []

for i in range(len(data)):

    k=[0]*len(all_ingd)

    for val in ingredients[i]:

        k[all_ingd.index(val.lower().strip())]=1

    ingd_vec.append(k)
data['ingd_vec']=ingd_vec
data.head()
similarity_dict={}

for i in range(len(data)):

    for j in range(0,len(data)):

        if i!=j:

            recipe_1 = data.loc[i,'name'].lower().strip()

            recipe_2 = data.loc[j,'name'].lower().strip()

            similarity_dict[recipe_1,recipe_2]=np.round(cosine_similarity([data.loc[i,'ingd_vec']],[data.loc[j,'ingd_vec']])[0,0],4)*100
recipe = [word.lower().strip() for word in data['name'].values]
reg_wise={}

for region in data.region.unique():

    reg_wise[region.lower()] = [recipe.lower().strip() for recipe in data[data['region']==region]['name'].values]
state_wise={}

for state in data.state.unique():

    state_wise[state.lower()] = [recipe.lower().strip() for recipe in data[data['state']==state]['name'].values]
def random_item_similarity(recipe_1,recipe_2):

    rec_1 = recipe_1.lower().strip()

    rec_2 = recipe_2.lower().strip()

    if rec_1 in recipe and rec_2 in recipe:

        print(f"Similarity of ingredients b/w {recipe_1} and {recipe_2} is {str(similarity_dict[rec_1,rec_2])}%.")

    else:

        print('Either or all of the food items provided do not belogn to the dataset.')
def reg_wise_similarity(recipe_1,recipe_2,region_1='-1',region_2='-1'):

    region_1 = region_1.lower().strip()

    region_2 = region_2.lower().strip()

    rec_1 = recipe_1.lower().strip()

    rec_2 = recipe_2.lower().strip()

    if region_1 in reg_wise.keys() and region_2 in reg_wise.keys():

        if (rec_1 in reg_wise[region_1] or rec_2 in reg_wise[region_1]) and (rec_1 in reg_wise[region_2] or rec_2 in reg_wise[region_2]):

            print(f"Similarity of ingredients b/w {recipe_1} and {recipe_2} is {str(similarity_dict[rec_1,rec_2])}%.")

        else:

            print('Sorry, Either or all of the food itmes provided do not belong to the the regions.')

    else:

        print('Sorry, One or both the regions provided do not belong to dataset.')
def state_wise_similarity(recipe_1,recipe_2,state_1='-1',state_2='-1'):

    state_1 = state_1.lower().strip()

    state_2 = state_2.lower().strip()

    rec_1 = recipe_1.lower().strip()

    rec_2 = recipe_2.lower().strip()

    if state_1 in state_wise.keys() and state_2 in state_wise.keys():

        if (rec_1 in state_wise[state_1] or rec_2 in state_wise[state_1]) and (rec_1 in state_wise[state_2] or rec_2 in state_wise[state_2]):

            print(f"Similarity of ingredients b/w {recipe_1} and {recipe_2} is {str(similarity_dict[rec_1,rec_2])}%.")

        else:

            print('Sorry, Either or all of the food item provided do not belong to the states.')

    else:

        print('Sorry, One or both the states provided do not belong to dataset.')
def similarity_func_format():

    print(f"Format of function for random item similarity : random_item_similarity('recipe_1','recipe_2')")

    print(f"Format of function for region-wise similarity : reg_wise_similarity('recipe_1','recipe_2','region_1','region_2')")

    print(f"Format of function for state-wise similarity  : state_wise_similarity('recipe_1','recipe_2','state_1','state_2')")
similarity_func_format()
random_item_similarity('gulab jamun','boondi')
reg_wise_similarity('gulab jamun','boondi','east','west')
state_wise_similarity('balu shahi','basundi','west bengal','gujarat')