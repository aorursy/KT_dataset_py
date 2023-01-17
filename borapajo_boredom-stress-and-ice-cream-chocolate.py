# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib as plt

import nltk

import re

import csv

import pandas as pd

food = pd.read_csv("../input/food.csv")



#create a list of all the comfort foods

with open("../input/food.csv", 'rU') as csvfile:

    csvreader = csv.reader(csvfile)

    comfort_food=[]

    for line in csvreader:

        comfort_food.append([line[7]])

        print (line[7])
#most common types of comfort food

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

new = [i for i in str(comfort_food).lower().split() if i not in stop]

Freq_dist_nltk=nltk.FreqDist(new)

Freq_dist_nltk.plot(30, cumulative = False)
#Let's look at main reasons for eating comfort food

with open("../input/food.csv", 'rU') as csvfile:

    csvreader = csv.reader(csvfile)

    comfort_food_reason=[]

    for line in csvreader:

        comfort_food_reason.append([line[8]])

        print (line[8])
#most common reasons for eating comfort food

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

new = [i for i in str(comfort_food_reason).lower().split() if i not in stop]

Freq_dist_nltk=nltk.FreqDist(new)

Freq_dist_nltk.plot(30, cumulative = False)
#Let's look at their current diet

with open("../input/food.csv", 'rU') as csvfile:

    csvreader = csv.reader(csvfile)

    current_diet=[]

    for line in csvreader:

        current_diet.append([line[11]])

        print (line[11])
#and the current diet is...

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

new = [i for i in str(current_diet).lower().split() if i not in stop]

Freq_dist_nltk=nltk.FreqDist(new)

Freq_dist_nltk.plot(30, cumulative = False)
#Let's look at eating changes since they have become students

with open("../input/food.csv", 'rU') as csvfile:

    csvreader = csv.reader(csvfile)

    eating_changes=[]

    for line in csvreader:

        eating_changes.append([line[13]])

        print (line[13])
#eating changes since they became students

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

new = [i for i in str(eating_changes).lower().split() if i not in stop]

Freq_dist_nltk=nltk.FreqDist(new)

Freq_dist_nltk.plot(30, cumulative = False)