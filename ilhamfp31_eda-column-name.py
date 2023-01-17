import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import operator

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
count_dash = {}

count_name = {}
for col in train.columns:

    if col not in ['id', 'target']:

        names = col.split('-')

        for name in names:

            if not name in count_name:

                count_name[name] = 1

            else:

                count_name[name] += 1



        if not len(names) in count_dash:

            count_dash[len(names)] = 1

        else:

            count_dash[len(names)] += 1
count_dash
sorted_count_name = sorted(count_name.items(), key=operator.itemgetter(1), reverse=True)

sorted_count_name[0:10]
# Saving to file so you can inspect the whole list

with open('count_name_train.txt', 'w') as f:

    for item in sorted_count_name:

        f.write("%s\n" % str(item))
len(sorted_count_name)
count_name_train = count_name
test.head()
count_dash = {}

count_name = {}
for col in test.columns:

    if col not in ['id']:

        names = col.split('-')

        for name in names:

            if not name in count_name:

                count_name[name] = 1

            else:

                count_name[name] += 1

        

        if not len(names) in count_dash:

            count_dash[len(names)] = 1

        else:

            count_dash[len(names)] += 1
count_dash
sorted_count_name = sorted(count_name.items(), key=operator.itemgetter(1), reverse=True)

sorted_count_name[0:10]
# Saving to file so you can inspect the whole list

with open('count_name_test.txt', 'w') as f:

    for item in sorted_count_name:

        f.write("%s\n" % str(item))
len(sorted_count_name)
set(count_name.keys()) - set(count_name_train.keys())
set(count_name_train.keys()) - set(count_name.keys())
# We can use only train column because it's exactly the same between test & train

print(set(train.columns) - set(test.columns))

print(set(test.columns) - set(train.columns))
column_1 = {}

column_2 = {}

column_3 = {}

column_4 = {}
for col in train.columns:

    if col not in ['id', 'target']:

        names = col.split('-')

        if not names[0] in column_1:

            column_1[names[0]] = 1

        else:

            column_1[names[0]] += 1



        if not names[1] in column_2:

            column_2[names[1]] = 1

        else:

            column_2[names[1]] += 1



        if not names[2] in column_3:

            column_3[names[2]] = 1

        else:

            column_3[names[2]] += 1



        if not names[3] in column_4:

            column_4[names[3]] = 1

        else:

            column_4[names[3]] += 1

        
col_list = [column_1, column_2, column_3, column_4]



for i in range(0,3):

    for j in range(i+1,4):

        print("Duplicate name between column {} and {}".format(i+1, j+1))

        print(set(col_list[i].keys()) & set(col_list[j].keys()))

print([x for x in train.columns if 'magic' in x.split('-')])
train['stealthy-chocolate-urchin-kernel']
train['bluesy-chocolate-kudu-fepid']