import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/cocktails-hotaling-co/hotaling_cocktails - Cocktails.csv")

dataset = data.fillna("No_value")

print(dataset.head())
dataset.describe()
dataset.info()
location = dataset["Location"].value_counts()

bartender = dataset['Bartender'].value_counts()

barCompany = dataset['Bar/Company'].value_counts()



print("Top 20 locations with highest number of cocktails:\n")

print(location[1:21])



print("\nTop 20 bartenders with highest number of cocktails:\n")

print(bartender[1:21])



print("\nTop 20 Bar/Company with highest number of cocktails:\n")

print(barCompany[1:21])
plt.figure(figsize = (10,6))

plt.plot(location[1:21]/687 * 100, 'bo')

plt.plot(location[1:21]/687 * 100, 'b-')

plt.xticks(rotation= 80)

plt.xlabel("Top 20 Places")

plt.ylabel("Number of cocktails in percentage")

plt.title("Cocktails made in each places")





plt.figure(figsize = (10,6))

plt.plot(bartender[1:21]/687*100, 'r-')

plt.plot(bartender[1:21]/687*100, 'ro')

plt.xticks(rotation= 80)

plt.xlabel("Top 20 Bartender's names")

plt.ylabel("Number of cocktails in percentage")

plt.title("Cocktails made by Bartenders")



plt.figure(figsize = (10,6))

plt.plot(barCompany[1:21]/687*100, 'go')

plt.plot(barCompany[1:21]/687*100, 'g-')

plt.xticks(rotation= 80)

plt.xlabel("Top 20 Bar/Company")

plt.ylabel("Number of cocktails in percentage")

plt.title("Cocktails made in Bar/Company")



plt.show()



ind = dataset['Ingredients']

ingredients = []



for element in ind:

    element = element.replace("(","")

    element = element.replace(")","")

    element = element.replace("*","")

    element = element.replace("\xa0"," ")

    ingredients.append([x.lstrip() for x in element.split(',')])

allWords = [] 



for element in ingredients:

    allWords.extend(list(map(lambda x: x,element)))

    

uniqueWord = []

finalAll = {}



for element in ingredients:

    uniqueWord.extend(list(filter(lambda x: x not in uniqueWord,element)))

        

for element in uniqueWord:

    finalAll[element] = allWords.count(element) 



IngredientFrame = pd.DataFrame(list(finalAll.items()), columns=['ingredients','count']).sort_values(by = 'count', ascending = False)

IngredientTop20 = IngredientFrame[0:21]



print(IngredientTop20)
plt.figure(figsize = (10,6))

plt.plot(IngredientTop20['ingredients'],IngredientTop20['count'], 'bo')

plt.plot(IngredientTop20['ingredients'],IngredientTop20['count'], 'b-')

plt.xlabel("Top 20 Ingredients")

plt.ylabel("Count of Ingredients")

plt.title("Ingredients Analytics")

plt.xticks(rotation= 80)

plt.show()
gar = [x.lower() for x in dataset['Garnish']]

garnish = []



garnish.extend(list(map(lambda x: re.split(",",x),gar)))



allWords = [] 



for element in garnish:

    for i in element:

        i = i.lstrip()

        allWords.append(i)

        

uniqueWords = []

finalAll = {}



for element in garnish:

    uniqueWords.extend(list(filter(lambda x: x not in uniqueWords,element)))



for element in uniqueWords:

    finalAll[element] = allWords.count(element)



dataframe = pd.DataFrame(list(finalAll.items()), columns=['garnish','count']).sort_values(by = 'count', ascending = False)

dataframe20 = dataframe[1:22]



plt.figure(figsize = (10,6))

plt.plot(dataframe20['garnish'],dataframe20['count'], 'bo')

plt.plot(dataframe20['garnish'],dataframe20['count'], 'b-')

plt.xlabel("Top 20 Garnish")

plt.ylabel("Count of Garnish")

plt.title("Garnish Analytics")

plt.xticks(rotation= 80)

plt.show()