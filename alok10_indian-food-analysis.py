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
%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data=pd.read_csv('../input/indian-food-101/indian_food.csv')
data.head()
data.tail()
data.shape
data.columns.values
east_food=data[:][data['region']=='East']
east_food
west_food=data[:][data['region']=='West']
west_food
region=set(data['region'])
region
central_food=data[:][data['region']=='Central']
central_food
north_food=data[:][data['region']=='North']
north_food
NE_food=data[:][data['region']=='North East']
NE_food
south_food=data[:][data['region']=='South']
south_food
veg_food=data[:][data['diet']=='vegetarian']
veg_food
veg_food.shape
non_veg_food=data[:][data['diet']=='non vegetarian']
non_veg_food
non_veg_food.shape
data[:][data['state']=='Uttar Pradesh']
states=set(data['state'])
len(states)
states
flavor=set(data['flavor_profile'])
len(flavor)
flavor
sweet=data[:][data['flavor_profile']=='sweet']
sweet
sweet.shape
sour=data[:][data['flavor_profile']=='sour']
sour
spicy=data[:][data['flavor_profile']=='spicy']
spicy
spicy.shape
bitter=data[:][data['flavor_profile']=='bitter']
bitter
sns.countplot(x='diet',data=data,hue='flavor_profile')
sns.countplot(x='course',data=data,hue='flavor_profile')
food_profile=[sweet.shape[0],sour.shape[0],bitter.shape[0],spicy.shape[0]]
food_profile
label=['sweet','sour','bitter','spicy']

col=['m','c','r','y','b']
plt.figure(figsize=(10,10))

plt.pie(food_profile,labels=label,colors=col,startangle=90,shadow=True,explode=(0.1,0.1,0.1,0.1),autopct='%1.1f%%')

plt.title('Indian Food Profile on basis of Taste')
plt.figure(figsize=(10,10))

plt.pie([veg_food.shape[0],non_veg_food.shape[0]],labels=['Veg','Non Veg'],colors=col,startangle=90,shadow=True,explode=(0.1,0.1),autopct='%1.1f%%')

plt.title('Indian Food Profile on basis of Menu')
plt.figure(figsize=(8,8))

plt.pie([south_food.shape[0],north_food.shape[0],west_food.shape[0],east_food.shape[0],NE_food.shape[0]],labels=['South','North','West','East','North_East'],colors=col,startangle=90,shadow=True,explode=(0.1,0.1,0.1,0.1,0.1),autopct='%1.1f%%')

plt.title('Indian Food Profile on basis of Region')
list(set(data['course']))
dessert=data[:][data['course']=='dessert']

main_course=data[:][data['course']=='main_course']

snack=data[:][data['course']=='snack']

starter=data[:][data['course']=='starter']
plt.figure(figsize=(8,8))

plt.pie([snack.shape[0],dessert.shape[0],starter.shape[0],main_course.shape[0]],labels=list(set(data['course'])),colors=col,startangle=90,shadow=True,explode=(0.1,0.1,0.1,0.1),autopct='%1.1f%%')

plt.title('Indian Food Profile on basis of Course')