# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

%matplotlib inline

menu = pd.read_csv("../input/nutrition-facts/menu.csv")

menu.head(10)
menu.describe(include='all')
len(menu)
menu.isnull().sum()
menu.isnull()
sns.heatmap(menu.isnull(),yticklabels=False,cbar=False,cmap='viridis')
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



### categorical scatter plots

plot = sns.swarmplot(x='Category',y='Sodium', data=menu)

plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)

plt.title('Sodium')

plt.show()



menu['Sodium'].describe()
menu['Sodium'].idxmax()
menu.at[82,'Item']
menu.columns
menu['Dietary Fiber'].idxmax()
menu.at[32,'Item']
subset = menu.loc[ [32] ,['Item','Serving Size', 'Protein','Total Fat','Dietary Fiber','Dietary Fiber (% Daily Value)'] ]

subset
subset['%Fiber'] = subset['Dietary Fiber']/434 * 100

subset
menu['Protein'].idxmax()
subset1 = menu.loc[ [82] ,['Item','Serving Size', 'Protein','Calories','Calories from Fat'] ]

subset1
subset1['Calories from Protein'] = subset['Protein']/4 

subset


df = menu[['Item','Serving Size','Calories','Protein']]



df.head()

df['Protein'].idxmax()
subset1
menu['Calories from Protein'] = menu['Protein']*4 

menu['Calories from Carbs'] = menu['Carbohydrates']*4

menu['%protein calorie'] = menu['Calories from Protein']/menu['Calories']*100

menu['%carbs calorie'] = menu['Calories from Carbs']/menu['Calories']*100

menu['%fat calorie'] = menu['Calories from Fat']/menu['Calories']*100

subset1 = menu.loc[ [82] ,['Item','Category','Serving Size','Calories','Protein','Carbohydrates',

                           'Calories from Protein','Calories from Fat','Calories from Carbs',

                          '%protein calorie','%carbs calorie','%fat calorie'] ]

subset1

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



### categorical scatter plots

plot = sns.swarmplot(x='Category',y='%fat calorie', data=menu)

plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)

plt.title('High Fat')

plt.show()
menu['Category'].unique()