#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd

import matplotlib.pyplot as plt



#===========================================================================

# read in the data 

# (2 month rolling leaderboard data, downloaded on the 3.V.2020)

#===========================================================================

titanic_lb = pd.read_csv('../input/titanic-publicleaderboarddata-4v2020/titanic-publicleaderboard.csv')



#===========================================================================

# make a frequency table

#===========================================================================

from collections import Counter

titanic_ft = Counter(titanic_lb["Score"])
plt.figure(figsize=(16,6))

plt.xlabel  ("Score")

plt.ylabel  ("Frequency")

plt.xlim((0.0,1.0))

plt.bar(titanic_ft.keys(), titanic_ft.values(), width=0.004)

plt.show()
plt.figure(figsize=(16,6))

plt.xlabel  ("Score")

plt.ylabel  ("Frequency")

plt.xlim((0.6,0.85))

plt.bar(titanic_ft.keys(), titanic_ft.values(), width=0.004)

plt.show()
# find the maximum value (i.e. most frequent score) 

# and its corresponding key

maximum = max(titanic_ft, key=titanic_ft.get)

# calculate the percentage of submissions that have this score

percentage_max_score = ((100/titanic_lb.shape[0])*titanic_ft[maximum])

print("Percentage of people with the most frequent score is:",

      str(round(percentage_max_score, 2)),"%")
# print the number of 'perfect' solutions

print("Number of 'perfect' (1.00000) submissions is: %i" % titanic_ft[1.0])
# sum the number of submissions with a score > 0.8

sum = 0

for key in titanic_ft:

    if key > 0.8:

        sum = sum + titanic_ft[key]

print("Number of submissions whose score is greater than 0.8 is:",sum)
# take away the 1.00000 bin

number_gt_8_correct = sum - titanic_ft[1.0]

print("less those with a perfect 1.00000 is:", number_gt_8_correct)
percentage_gt_eight = ((100/titanic_lb.shape[0])*number_gt_8_correct)

print("Submissions with a score greater than 0.8 are in the top", 

      str(round(percentage_gt_eight, 2)),"%")