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
# import relevant packages



import csv as csv # for reading and writing csv files



# open the csv file in to a Python object

csv_file_object = csv.reader(open('../input/train.csv'))



header = next(csv_file_object) # next() command skips the first

                                # line which is a header

print(header)



data = []  # create a variable called 'data'



# run through aech row in the csv file, adding each row

# to the data variable

for row in csv_file_object:

    data.append(row)

    

# covert from a list to an array

# NB each item is currently a STRING

data = np.array(data)
print(data) # data is an array with just values (no header)

            # values stored as strings
print(data[0]) # see first row

print(data[-1]) # and last row

print(data[0,3]) # see 1st row, 4th column
# if want specific column, e.g. gender column:

# data[0::,4]

# will need to convert strings to floats to do calculations

# e.g Pclass into floats: data[0::,2].astype(np.float)



# The size() function counts how many elements are in

# the array and sum() sums the elements in the array



# calculate the proportion of survivors on the Titanic

number_passengers = np.size(data[0::,1].astype(np.float))

number_survived = np.sum(data[0::,1].astype(np.float))

proportion_survivors = number_survived / number_passengers



print(proportion_survivors)
# determine number of females and males that survived

women_only_stats = data[0::,4] == "female" # finds where all

                                           # elements of the gender

                                           # column that equal "female"

men_only_stats = data[0::,4] != "female" # finds where all the

                                         # elements do not equal

                                         # female (i.e. male)

# use these new variables as a "mask" on our original data

# to get stats on only men and only women
# Using the index from above select femails and males separately

women_onboard = data[women_only_stats,1].astype(np.float)

men_onboard = data[men_only_stats,1].astype(np.float)



# Find the proportion of women and proportion of men that survived

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)



print('Proportion of women who survived is %s' % proportion_women_survived)

print('Proportion of men who survived is %s' % proportion_men_survived)
# read in the test.csv and skip the header line

test_file = open('../input/test.csv')

test_file_object = csv.reader(test_file)

header = next(test_file_object)
# open a pointer to a new file so we can write to it

# (the file does not exit yet)

prediction_file = open("genderbasedmodel.csv", "wt", newline='\n')

prediction_file_object = csv.writer(prediction_file)
# read the test file row by row

# see if male or femail and write survival prediction to a new file

prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:

    if row[4] == 'female':

        prediction_file_object.writerow([row[0], '1']) # predict 1

    else:

        prediction_file_object.writerow([row[0], '0']) # predict 0

test_file.close()

prediction_file.close()
testfile = open("jamietest.csv", "w")

testfile.close()