# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = check_output(["ls", "../input"]).decode("utf8")

# Any results you write to the current directory are saved as output.
import numpy as np

import io

import pylab

from sklearn.svm import SVC

from sklearn import preprocessing

import sys

import csv as csv
csv_file_object = csv.reader(open("../input/UCI_Credit_Card.csv"))
header = next(csv_file_object)
data=[]
for row in csv_file_object:

    data.append(row)
data = np.array(data)
data
print(data[0])
print(data[0::,2])
number_people = np.size(data[0::,1].astype(np.float))

print(number_people)
number_defaulted = np.sum(data[0::,24].astype(np.float))

print("Total Ratio of People Defaulting: %f" % ( number_defaulted / number_people))
print(number_defaulted)
women_only_stats = data[0::,2]=="2"

np.sum(women_only_stats)
men_only_stats = data[0::,2]=="1"

np.sum(men_only_stats)
women_defaulting = data[women_only_stats,24].astype(np.float)

np.sum(women_defaulting)
men_defaulting = data[men_only_stats,24].astype(np.float)

np.sum(men_defaulting)
proportion_women_defaulting = np.sum(women_defaulting) / np.size(women_only_stats)
proportion_men_defaulting = np.sum(men_defaulting) / np.size(men_only_stats)
print('The proportion of women defaulting is %f' % (proportion_women_defaulting))

print('The proportion of men defaulting is %f' % (proportion_men_defaulting))