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
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv("../input/food-preferences/Food_Preference.csv")
df.head(2)
data = df #creating copy of the data



columns_with_null = data.isnull().sum()

columns_with_null = columns_with_null[columns_with_null != 0]
columns_with_null
nan_entries = data.isnull()

row_has_nan = nan_entries.any(axis = 1)

data[row_has_nan]
print(list(data['Nationality'].unique()))
a4_dims = (21, 6)

fig, ax = plt.subplots(1,2,figsize=a4_dims)

ax[0].set(title = 'Distribtion Including Indian Nationality')

ax[1].set(title = 'Distribution Excluding Indian Nationality')

sns.barplot(x = data['Nationality'].value_counts(),y= data['Nationality'].value_counts().index, ax = ax[0])

sns.barplot(x = data['Nationality'][data['Nationality'] != 'Indian'].value_counts(),y= data['Nationality'][data['Nationality'] != 'Indian'].value_counts().index, ax = ax[1])
rowsWithIndianNationality = float(data['Nationality'][data['Nationality'] == "Indian"].value_counts())

totalRows = float(data['Nationality'].value_counts().sum())

perc = (rowsWithIndianNationality)/(totalRows)*100



print("The % of data belonging to Indian Nationality is:", perc,"%")

dims = (20, 5)

fig, ax = plt.subplots(1,3,figsize=dims)

ax[0].set(title = 'Distribution of Food')

ax[1].set(title = 'Distrubition of Juice')

ax[2].set(title = 'Distribution of Dessert')

#ax[0].set_xticklabels(ax[0].get_yticklabels(), rotation=45)

#ax[1].set_xticklabels(ax[1].get_yticklabels(), rotation=90)

#ax[2].set_xticklabels(ax[2].get_yticklabels(), rotation=90)

sns.barplot(x = data[data['Nationality'] == "Indian"].Food.value_counts(), y = data.Food.value_counts().index, ax = ax[0])

sns.barplot(x = data[data['Nationality'] == "Indian"].Juice.value_counts(), y = data.Juice.value_counts().index, ax = ax[1])

sns.barplot(x = data[data['Nationality'] == "Indian"].Dessert.value_counts(), y = data.Dessert.value_counts().index, ax = ax[2])
dims = (20, 5)

fig, ax = plt.subplots(1,3,figsize=dims)

ax[0].set(title = 'Gender-wise count of Food Choice')

ax[1].set(title = 'Gender-wise count of Juice Choice')

ax[2].set(title = 'Gender-wise count of Dessert Choice')



#visualizing the above

sns.countplot('Food', hue = 'Gender', data = data[data['Nationality'] == "Indian"], ax = ax[0])

sns.countplot('Juice', hue = 'Gender', data = data[data['Nationality'] == "Indian"], ax = ax[1])

sns.countplot('Dessert', hue = 'Gender', data = data[data['Nationality'] == "Indian"], ax = ax[2])
sns.distplot(data['Age'], color = 'g')
sns.FacetGrid(data[data['Nationality'] == "Indian"], col = 'Food', row = 'Gender').map(sns.distplot, 'Age')
sns.FacetGrid(data[data['Nationality'] == "Indian"], col = 'Juice', row = 'Gender').map(sns.distplot, 'Age')
sns.FacetGrid(data[data['Nationality'] == "Indian"], col = 'Dessert', row = 'Gender').map(sns.distplot, 'Age')
dims = (10, 15)

fig, ax = plt.subplots(3,2,figsize=dims)

ax[0][0].set(title = 'Males Only (Food Choice vs Age)')

ax[0][1].set(title = 'Females Only(Food Choice vs Age)')



sns.swarmplot(y = data['Age'][data.Gender == 'Male'], x = data['Food'][data.Gender == 'Male'], ax = ax[0][0])

sns.swarmplot(y = data['Age'][data.Gender == 'Female'], x = data['Food'][data.Gender == 'Female'], ax = ax[0][1])



ax[1][0].set(title = 'Males Only (Juice Choice vs Age)')

ax[1][1].set(title = 'Females Only(Juice Choice vs Age)')



sns.swarmplot(y = data['Age'][data.Gender == 'Male'], x = data['Juice'][data.Gender == 'Male'], ax = ax[1][0])

sns.swarmplot(y = data['Age'][data.Gender == 'Female'], x = data['Juice'][data.Gender == 'Female'], ax = ax[1][1])



ax[2][0].set(title = 'Males Only (Dessert Choice vs Age)')

ax[2][1].set(title = 'Females Only (Dessert Choice vs Age)')



sns.swarmplot(y = data['Age'][data.Gender == 'Male'], x = data['Dessert'][data.Gender == 'Male'], ax = ax[2][0])

sns.swarmplot(y = data['Age'][data.Gender == 'Female'], x = data['Dessert'][data.Gender == 'Female'], ax = ax[2][1])
dims = (20, 5)

fig, ax = plt.subplots(1,3,figsize=dims)



ax[0].set(title = 'Food Choice vs Age')

sns.swarmplot(y = data['Age'], x = data['Food'], ax = ax[0])



ax[1].set(title = 'Juice Choice vs Age')

sns.swarmplot(y = data['Age'], x = data['Juice'], ax = ax[1])



ax[2].set(title = 'Dessert Choice vs Age')

sns.stripplot(y = data['Age'], x = data['Dessert'], ax = ax[2])
dims = (20, 5)

fig, ax = plt.subplots(1,2,figsize=dims)

ax[0].set(title = 'When Choice is Fresh Juice')

sns.countplot('Dessert', hue = 'Food', data = data[data.Juice == 'Fresh Juice'], ax = ax[0])



ax[1].set(title = 'When Choice is Carbonated Drinks')

sns.countplot('Dessert', hue = 'Food', data = data[data.Juice == 'Carbonated drinks'], ax = ax[1])