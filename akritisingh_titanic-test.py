# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



csv_file_object = csv.reader(open('../input/train.csv', 'rt')) 

header = next(csv_file_object)  # The next() command just skips the 

                                 # first line which is a header

train_data=[]                          # Create a variable called 'data'.

for row in csv_file_object:      # Run through each row in the csv file,

    train_data.append(row)             # adding each row to the data variable

train_data = np.array(train_data) 	         # Then convert from a list to an array

			         # Be aware that each item is currently

                                 # a string in this format

        

survive_col = train_data[::,1]

class_col = train_data[::,2]

name_col = train_data[::,3]

gender_col = train_data[::,4]

age_col = train_data[::,5]

    
#########finding percentage of survived ppl who were female

fem = gender_col == "female"

sur = survive_col == "1"

fem_sur = np.logical_and(fem, sur)

num_survived = np.count_nonzero(sur)

percent_surviving_females = np.count_nonzero(fem_sur)/num_survived

print(percent_surviving_females)





    
#####social class

first_class_survivors = np.logical_and(class_col == "1", sur)

second_class_survivors = np.logical_and(class_col == "2", sur)

third_class_survivors = np.logical_and(class_col == "3", sur)



p1 = np.count_nonzero(first_class_survivors)/num_survived

p2 = np.count_nonzero(second_class_survivors)/num_survived

p3 = np.count_nonzero(third_class_survivors)/num_survived



fem1 = np.count_nonzero(np.logical_and(first_class_survivors, fem)) #num of first class female survivors



print("Out of all first class survivors, " + str(fem1/np.count_nonzero(first_class_survivors) * 100) + "% were female.")

print("Out of all female survivors, " + str(fem1/np.count_nonzero(fem_sur) * 100) + "% of them were first class.")



print("Out of all survivors, " + str(p1) + "% were first class")

print("Out of all survivors, " + str(p2) + "% were second class")

print("Out of all survivors, " + str(p3) + "% were third class")
############names

surnames = []

titles = []



for name in name_col:

    split_names = name.split()

    surnames.append(split_names[0])

    for i in range(len(split_names)):

        if split_names[i][-1:] == ",":

            titles.append(split_names[i+1])



surname_sur = {}

title_sur = {}

count = 0

for name in surnames:

    if name not in surname_sur.keys():

        surname_sur[name] = []

    surname_sur[name].append(sur[count])

    count+=1

    

count = 0

for title in titles:

    if title not in title_sur.keys():

        title_sur[title] = []

    title_sur[title].append(sur[count])

    count+=1

    

    

majority_survived = 0

families = 0

for key, val in surname_sur.items():

    if len(val) > 1:

        if np.count_nonzero(np.array(val)) >= len(val)/2:

            majority_survived+=1

        families+=1

        

print ("Fams: " + str(families) + " Majority survived: " + str(majority_survived))

            



##for each name, we should look at the corresponding number of sbiglins +

          




for key, val in title_sur.items():

    title_sur[key] = np.count_nonzero(np.array(val))/len(val)



print(title_sur)