'''

Created on Jun 26, 2017



Top 5 ingredients as pie chart



@author: HSCC_HsinWei

'''



import json

import io

import ast

from matplotlib import pyplot as plt 

import operator



# Reading the data

with io.open('../input/train.json', 'r',encoding='utf-8') as train_file:

    train_data = json.load(train_file)



for i in range(len(train_data)):

    train_data[i]=ast.literal_eval(json.dumps(train_data[i]))



keyIngredientsValAmount={}

for i in range(len(train_data)):

    for ingredient in train_data[i]['ingredients']:

        if ingredient in list(keyIngredientsValAmount.keys()):

            keyIngredientsValAmount[ingredient]=keyIngredientsValAmount[ingredient]+1

        else:

            keyIngredientsValAmount[ingredient]=1

            

#calculate all amount of ingredient

sumOfIngredients=sum(keyIngredientsValAmount.values())



#sort by amount of ingredient

sortedByAmount = sorted(keyIngredientsValAmount.items(), key=operator.itemgetter(1),reverse=True)



# plot Pie chart with top "XX" percentage 

top=5

plt.figure(figsize=(6,9))

labels = []

sizes = []

for i in range(top):

    labels.append(sortedByAmount[i][0])

    sizes.append(sortedByAmount[i][1]/sumOfIngredients)

 

labels.append('other')

sizes.append(1-sum(sizes))

 

patches,l_text,p_text = plt.pie(sizes,labels=labels,

                                labeldistance = 1.1,autopct = '%3.1f%%',shadow = False,

                                startangle = 90,pctdistance = 1)

 

plt.axis('equal')

plt.legend()

plt.savefig('5_most_used_ingredients.png')