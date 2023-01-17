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

dataset = pd.read_csv('../input/nutrition-facts/menu.csv')
calories_data = dataset.iloc[:,0:5]
calories_data['Total Calories'] = calories_data['Calories']+calories_data['Calories from Fat']
calories_by_category = calories_data.groupby('Category').mean().sort_values(by='Total Calories')
calories_by_category

calories_by_category.iloc[:,:2].plot(kind='bar', stacked=True)
plt.title('Total Calories Average by Category')
plt.ylabel('Total Calories')
plt.show()
sandwich_data = dataset[dataset['Item'].str.contains('Sandwich')]
sandwich_data = sandwich_data.iloc[:,[1,6,8,11,13,15,17,20,21,22,23]]
sandwich_data['Total Daily Value'] = sandwich_data.sum(axis=1)
sandwich_data['Chicken Type'] = sandwich_data['Item'].str.extract("(Crispy|Grilled)")
sandwich_data['Item Type'] = sandwich_data['Item'].str.extract("(Classic|Club|Ranch|Bacon|Southern)")
sandwich_data



sandwich_datax = dataset[dataset['Item'].str.contains('Sandwich')]
#sandwich_datax = sandwich_datax.iloc[:,[3,5,8,11,13,15,17,20,21,22,23]]
sandwich_datax = sandwich_datax.iloc[:,[1,3,5,7,10,12,14,16,18,19]]
sandwich_datax['Chicken Type'] = sandwich_datax['Item'].str.extract("(Crispy|Grilled)")
sandwich_datax
sandwich_datax['Total Calories'] = sandwich_datax['Calories']+sandwich_datax['Total Fat']+sandwich_datax['Saturated Fat']+sandwich_datax['Sodium']+sandwich_datax['Carbohydrates']+sandwich_datax['Dietary Fiber']+sandwich_datax['Sugars']+sandwich_datax['Protein']+sandwich_datax['Cholesterol']

nutrition_by_item = sandwich_datax.groupby('Item').mean().sort_values(by='Item')
nutrition_by_item

nutrition_by_item.iloc[:,[0,1,2,3,4,5,6,7,8]].plot(kind='bar', stacked=True)
#nutrition_by_item.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13]].plot(kind='bar', stacked=True)
plt.title('Grilled vs Crispy')
plt.ylabel('Total Nutrition')
plt.figure(figsize=(10,7))
plt.show()
nutrition_by_type = sandwich_datax.groupby('Chicken Type').mean().sort_values(by='Chicken Type')
nutrition_by_type

nutrition_by_type.iloc[:,[0,1,2,3,4,5,6,7,8]].plot(kind='bar', stacked=True)
#nutrition_by_item.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13]].plot(kind='bar', stacked=True)
plt.title('Grilled vs Crispy')
plt.ylabel('Total Nutrition')
plt.figure(figsize=(10,7))
plt.show()
plt.figure(figsize=(10,7))
sns.barplot(sandwich_data['Chicken Type'], sandwich_data['Total Daily Value'], hue=sandwich_data['Item'])
plt.show()
plt.figure(figsize=(10,7))
sns.barplot(sandwich_data['Item Type'], sandwich_data['Total Daily Value'], hue=sandwich_data['Chicken Type'])
plt.show()
egg_data = dataset[dataset['Item'].str.contains('Egg')]
#sandwich_datax = sandwich_datax.iloc[:,[3,5,8,11,13,15,17,20,21,22,23]]
egg_data = egg_data.iloc[:,[1,3,5,7,10,12,14,16,18,19]]
egg_data['Egg Type'] = egg_data['Item'].str.extract("(Egg Whites|Egg)")
egg_data['Egg Type'] = egg_data['Egg Type'].map({'Egg Whites': "Egg Whites", 'Egg': "Whole Eggs"})

egg_data
nutrition_by_type_egg = egg_data.groupby('Egg Type').mean().sort_values(by='Egg Type')
nutrition_by_type_egg
nutrition_by_type_egg.iloc[:,[0,1,2,3,4,5,6,7,8]].plot(kind='bar', stacked=True)
#nutrition_by_item.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13]].plot(kind='bar', stacked=True)
plt.title('Whole Eggs vs Egg Whites')
plt.ylabel('Total Nutrition')
plt.figure(figsize=(10,7))
plt.rcParams["figure.figsize"] = [10,7]
plt.show()