# hi guys! this is my first project. below I have some notes for myself (just because i'm still in the learning process.)

# any advice would be great!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STEP 1: SET UP IMPORTS



import csv

# from kagle, you first import (always)

import matplotlib.pyplot as plt

# from online to create a scatterplot





file = open('../input/top50spotify2019/top50.csv', encoding = 'latin')

# open the file using the formula above



csv_reader = csv.reader(file, delimiter=',')

# this allows me to process it as csv data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STEP 2: CREATE YOUR LISTS 



bpm = []

danceability = []

# creating our new list for each axis



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STEP 3: ADD DATA TO LISTS

for row in csv_reader:

    if row[0] == '':

    # make sure it's brackets cause list

        continue

    bpm.append(int(row[4]))

    danceability.append(int(row[6]))

    # assigning/adding/appending to our list

    

    

    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STEP 4: LAYOUT PLANNING

x = bpm; y = danceability

plt.scatter(x,y)

plt.title("Correlation between BPM and Danceability")

plt.xlabel("Beats per Minute")

plt.ylabel("Danceability")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# STEP 5: DISPLAY GRAPH

plt.show()

# don't forget the parentheses



# Conclusion: there is no visible correlation between BPM and Danceability.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Mnemonic to remember: S.C.A.L.D