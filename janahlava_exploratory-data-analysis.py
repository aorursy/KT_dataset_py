# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import re, csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data= pd.read_csv("../input/FoodFacts.csv", low_memory=False)
# Creating a copy of the original dataset as sub. All experiments will be done on sub
sub=data

# Print the column headers/headings
names=data.columns.values
print (names)
# Set the rows with missing data as -1 using the fillna()
sub=sub.fillna(-1)
def printFoodWithContent(content_list, log_disjunction = True, print_list = []):
    print (content_list)
    print_i = {}
    for content, values in content_list.items():
        i = 0
        for v in sub[content]:
            if v >= 0:
                if v >= values[0] and v <= values[1]:
                    if i not in print_i or log_disjunction:
                        print_i[i] = True
                elif not log_disjunction: # conjunction
                    print_i[i] = False
            i += 1
    for i in range(len(sub)):
        if not i in print_i or not print_i[i]:
            continue
        print ()
        for k in sub.keys():
            if k == 'categories' or k == 'product_name' or k == 'generic_name' or k == 'brands' or k in content_list or k in print_list:
                if sub[k][i] != -1:
                    print (k, sub[k][i])

# booze
print ('BOOOZE')
content_list = {}
content_list['alcohol_100g'] = [0.5, 100]
printFoodWithContent(content_list, print_list = ['countries'])

# energy drinks 
print ('KICKERS')
content_list = {}
content_list['caffeine_100g'] = [3, 100]
content_list['taurine_100g'] = [0.3, 100]
printFoodWithContent(content_list, print_list = ['energy_100g', 'quantity'])

# FODMAPs (food to avoid in low FODMAP diet: fructose, lactose, fructans, galactans, polyols) 
print ('FODMAP')
content_list = {}
content_list['fructose_100g'] = [5, 100]
content_list['lactose_100g'] = [1, 100]
content_list['polyols_100g'] = [30, 1000]
printFoodWithContent(content_list, print_list = ['polyols_100g', 'sugars_100g', 'sucrose_100g', 'glucose_100g', 'fructose_100g', 'lactose_100g', 'maltose_100g', 'ingredients_text'])

print ('FATTIES BAD')
content_list = {}
content_list['trans_fat_100g'] = [1, 100]
printFoodWithContent(content_list, print_list = ['energy_100g', 'quantity', 'saturated_fat_100g', 'fat_100g', 'monounsaturated_fat_100g', 'polyunsaturated_fat_100g', 'omega_3_fat_100g', 'omega_6_fat_100g', 'omega_9_fat_100g'])

print ('FATTIES GOOD')
content_list = {}
content_list['monounsaturated_fat_100g'] = [0.01, 10]
content_list['polyunsaturated_fat_100g'] = [0.01, 10]
content_list['omega_3_fat_100g'] = [1, 100]
content_list['omega_6_fat_100g'] = [1, 100]
content_list['omega_9_fat_100g'] = [1, 100]
content_list['trans_fat_100g'] = [0, 0.1]
printFoodWithContent(content_list, log_disjunction = False, print_list = ['energy_100g', 'quantity', 'saturated_fat_100g', 'fat_100g'])

