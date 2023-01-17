import csv as csv

import numpy as np
# Open up the csv file in to a Python object

csv_file_object = csv.reader(open('../input/train.csv', 'rt'))

header = next(csv_file_object)

data = []

for row in csv_file_object:

    data.append(row)

print(data[0])

data = np.array(data)

print(data[0])
print(data[0])

print(data[-1])

print(data[0,3])
# The size() function counts how many elements are in

# in the array and sum() (as you would expects) sums up

# the elements in the array.



Number_Survived = np.sum(data[0::,1].astype(np.float))

Total_Passengers = np.size(data[0::,1].astype(np.float))

Survived_Stat = Number_Survived / Total_Passengers

print(Survived_Stat)
# This finds where all 

# the elements in the gender

# column that equals “female”

# This finds where all the 

# elements do not equal 

# female (i.e. male)

# Mask creation



women = data[0::,4] == "female"

men = data[0::,4] != "female"

  
# Using the index from above we select the females and males separately



women_stat = data[women,1].astype(np.float)

men_stat = data[men,1].astype(np.float)

women_survived = np.sum(women_stat) / np.size(women_stat)

men_survived = np.sum(men_stat) / np.size(men_stat)

print('Women who survived is %s' %women_survived,'\n','Men who survived is %s' %men_survived)


test_file = open('../input/test.csv', 'rt')

test_file_object = csv.reader(test_file)

header = next(test_file_object)

prediciton_file = open('genderbasedmodel.csv', 'wt')

prediction_file_object = csv.writer(prediciton_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for row in test_file_object:

    if row[3] == "female":

        prediction_file_object.writerow([row[0], '1'])

    else:

        prediction_file_object.writerow([row[0], '0'])

test_file.close()

prediciton_file.close()