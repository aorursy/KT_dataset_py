# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import csv

import os

csv_file = os.listdir("../input")

print(csv_file)



# Any results you write to the current directory are saved as output.
# Read input file



stats = []

with open('../input/Pokemon.csv','rt')as f:

    data = csv.reader(f)

    for row in data:

        stats.append(row)

        

key = stats[0]

del stats[0]



print("Key -", key)

print('\n')

print("Stats -", stats)
# Pokemon count



print("Total -", len(stats))

leg_stats = []

for stat in stats:

    if stat[12] == 'True':

        leg_stats.append(stat)   

print("Legendary -", len(leg_stats))

print("Ratio -", len(leg_stats)/len(stats))
# Analyse pokemon types



types = []

for stat in stats:

    if stat[2] is not '':

        types.append(stat[2])

    if stat[3] is not '':

        types.append(stat[3])



types = set(types)

print("Types -", len(types), types)



print('\n')



# 1) Analyse if having dual type affects legendary status

num = 0; dem = 0

for stat in stats:

    if stat[3] == '':

        dem += 1

for stat in leg_stats:

    if stat[3] == '':

        num += 1

print("Dual Types -", num/dem, num, dem)



print('\n')



# Analyse if having a particular type1 affects legendary status

for type_temp in types:

    num = 0; dem = 0

    for stat in stats:

        if stat[2] == type_temp:

            dem += 1

    for stat in leg_stats:

        if stat[2] == type_temp:

            num += 1

    print("Type 1 -", type_temp, num/dem, num, dem)



print('\n')  



# Analyse if having a particular type2 affects legendary status

for type_temp in types:

    num = 0; dem = 0

    for stat in stats:

        if stat[3] == type_temp:

            dem += 1

    for stat in leg_stats:

        if stat[3] == type_temp:

            num += 1

    print("Type 2 -", type_temp, num/dem, num, dem)



print('\n')  



# Analyse if having a particular type affects legendary status

for type_temp in types:

    num = 0; dem = 0

    for stat in stats:

        if stat[2] == type_temp or stat[3] == type_temp:

            dem += 1

    for stat in leg_stats:

        if stat[2] == type_temp or stat[3] == type_temp:

            num += 1

    print("Any Type -", type_temp, num/dem, num, dem)



print('\n')    

    

# Analyse if having only a particular type affects legendary status

for type_temp in types:

    num = 0; dem = 0

    for stat in stats:

        if stat[2] == type_temp and stat[3] == '':

            dem += 1

    for stat in leg_stats:

        if stat[2] == type_temp and stat[3] == '':

            num += 1

    print("Only Type -", type_temp, num/dem, num, dem)



print('\nFor Comparison:')

print("Ratio -", len(leg_stats)/len(stats))

print("Legendary -", len(leg_stats))

print("Total -", len(stats))  
# Analyse pokemon generations



stats_gen = []

leg_stats_gen = []



for i in range(6):

    gen = i + 1

    stats_temp = []

    leg_stats_temp = []

    for stat in stats:

        if stat[11] == str(gen):

            stats_temp.append(stat)

    for stat in leg_stats:

        if stat[11] == str(gen):

            leg_stats_temp.append(stat)

            

    stats_gen.append(stats_temp)

    leg_stats_gen.append(leg_stats_temp)

    

for i in range(6):

    gen = i + 1

    print(gen, '-', len(leg_stats_gen[i])/len(stats_gen[i]), len(leg_stats_gen[i]), len(stats_gen[i]))
# Analyse pokemon combat stats for each generation



stat_names = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



for i in range(6):

    gen = i + 1

    print('Generation -', gen)

    for j in range(7):

        stat_name = stat_names[j]

        idx = j + 4

        values = []

        leg_values = []

        

        for stat in stats_gen[i]:

            if stat[12] == 'False':

                values.append(int(stat[idx]))

        for stat in leg_stats_gen[i]:

            leg_values.append(int(stat[idx]))

            

        print('\tCombat stat -', stat_name, '(Min, Avg, Max)')

        print('\t\tAll pokemon -', min(values), sum(values)/len(values), max(values))

        print('\t\tLegendary pokemon -', min(leg_values), sum(leg_values)/len(leg_values), max(leg_values))