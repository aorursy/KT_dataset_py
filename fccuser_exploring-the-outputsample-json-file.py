# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import json

with open('../input/outputsample.json','r') as f_in:

    datasample = json.load(f_in)

print(len(datasample))
#a quick histogram of how many exercises completed per student

import collections

import operator

completed_exer = collections.defaultdict(int)

for u in datasample:

    completed_exer[len(datasample[u])] = completed_exer[len(datasample[u])] + 1



vals1 = sorted(completed_exer.items(), key=operator.itemgetter(1), reverse=True)[0]

print("max num of exercises completed by most of the students : {0}\n (num of students that completed them: {1})".format(vals1[0], vals1[1]))



print('\n')



vals2 = sorted(completed_exer.items(), key=operator.itemgetter(1), reverse=True)[-1]

print("max num of exercises completed by less of the students: {0}\n (num of students that completed them: {1})".format(vals2[0], vals2[1]))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=False)



x = [x for x in range(len(completed_exer))]

xlabels = [xi if xi%10==0 else "" for xi in x]

y = []



for xi in x:

    if xi in list(completed_exer.keys()):

        y.append(completed_exer[xi])

    else:

        y.append(0)



plt.figure(figsize=(15,8))



ax = sns.barplot(x=x, y=y, color='steelblue')

ax.set(xticklabels=xlabels)

ax.figure.suptitle("Total of completed exercises vs Number of students that completed them", fontsize = 16)



plt.ylabel('total completed exercises', fontsize=10)

plt.xlabel('number of students', fontsize=10)

plt.show()