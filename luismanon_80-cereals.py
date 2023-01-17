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
#importing data science modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cereals = pd.read_csv("../input/80-cereals/cereal.csv")
print(cereals)
cereals.info()
cereals.head(10)
cereals.tail(10)
print(cereals.loc[0,:])
cereals.loc[:,:]
cereals.shape
cereals.size
cereals.index
cereals.columns
print(cereals.ndim)
cereals['mfr'].astype('category')
#Memory used by category data type and by normal object data type
cereals['mfr'].memory_usage
cereals['mfr'].astype('category').memory_usage
#find mean of proteins column
cereals['protein'].mean()
#printing name of cereal having more than 13 units of sugar
print(cereals['name'][cereals['protein']>13])
#Homework 3
#sugar section
for idx in cereals.index:
    cereal_id =  cereals.loc[idx,'name']
    sugar_num =  cereals.loc[idx,'sugars']
    cups_num = cereals.loc[idx,'cups']
    #compute sugar per once
    cereals.loc[idx, 'SugarPerOnce'] = sugar_num * cups_num

    
#calories section
#1 gram of carbo = 4 cal
#1 gram of fat = 9 cal
#1 protein = 4
for idx in cereals.index:
    #1 gram of sugar = 200 calories
    cereal_id =  cereals.loc[idx,'name']
    calories_num =  cereals.loc[idx,'calories']
    carbo = cereals.loc[idx,'carbo'] * 4
    protein =  cereals.loc[idx,'protein'] * 4
    fat =  cereals.loc[idx,'fat'] * 9
    #compute sugar per once
    cereals.loc[idx, 'CaloriesPerGram'] = calories_num * (carbo+protein+fat)
    

def findLeastProduct(c):
    product =""
    once = 9999999
    for x in c.index:
        if once > c.loc[x,'SugarPerOnce']:
            once = c.loc[x,'SugarPerOnce']
            product =  c.loc[x,'name']
    return product

#highest calories per gram
def findHighestCalorieOnGramProduct(c):
    product =""
    gram = 0
    for x in c.index:
        if gram < c.loc[x,'CaloriesPerGram']:
            gram = c.loc[x,'CaloriesPerGram']
            product =  c.loc[x,'name']
    return product, gram
#lowest calories per gram
def findLowestCalorieOnGramProduct(c):
    product =""
    gram = 99999
    for x in c.index:
        if gram > c.loc[x,'CaloriesPerGram']:
            gram = c.loc[x,'CaloriesPerGram']
            product =  c.loc[x,'name']
    return product, gram

#find the highest ratings in cereal
def TopFiveHighestRatings(c):
    rating_arr =  c["rating"]
    print(np.argsort(-rating_arr))
    ary =  np.argsort(-rating_arr)
    counter=1
    for index in  ary.index:
        if counter <= 5:
            print("product",c.loc[index,'name'],"rating = ",c.loc[index,'rating'])
            counter+=1
     
#the product with the least amount of sugar per once
prod = findLeastProduct(cereals)
print(prod)
#print the average of sugar per once
print("average %.2f"%cereals['SugarPerOnce'].mean())
#highest product with calories per gram
prod, cal=findHighestCalorieOnGramProduct(cereals)
print("Highest calories per gram prod,",cal," Calories Per Gram")
prod, cal=findLowestCalorieOnGramProduct(cereals)
print("Lowest calories per gram prod,",cal," Calories Per Gram")
TopFiveHighestRatings(cereals) 

