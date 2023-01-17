# The first thing to do is to import the relevant packages

# that I will need for my script, 

# these include the Numpy (for maths and arrays)

# and csv for reading and writing csv files

# If i want to use something from this I need to call 

# csv.[function] or np.[function] first



import csv as csv 

import numpy as np

import random



# Open up the csv file in to a Python object

csv_file_object = csv.reader(open('../input/train.csv', 'r')) 

header = next(csv_file_object)  # The next() command just skips the 

                                 # first line which is a header

data = []                        # Create a variable called 'data'.

for row in csv_file_object:      # Run through each row in the csv file,

    data.append(row)             # adding each row to the data variable

data = np.array(data) 	         # Then convert from a list to an array

			                     # Be aware that each item is currently

                                 # a string in this format
# So we add a ceiling

fare_ceiling = 40

# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling

data[ data[:,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0



fare_bracket_size = 10

number_of_price_brackets = fare_ceiling // fare_bracket_size



# I know there were 1st, 2nd and 3rd classes on board

number_of_classes = 3



# But it's better practice to calculate this from the data directly

# Take the length of an array of unique values in column index 2

number_of_classes = len(np.unique(data[:,2])) 



# Initialize the survival table with all zeros

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))
for i in range(number_of_classes):             #loop through each class

  for j in range(number_of_price_brackets):    #loop through each price bin

    women_only_stats = data[                          \

                         (data[:,4] == "female")      \

                       & (data[:,2].astype(np.float)  \

                             == i+1)                  \

                       & (data[:,9].astype(np.float)  \

                            >= j*fare_bracket_size)   \

                       & (data[:,9].astype(np.float)  \

                            < (j+1)*fare_bracket_size)\

                          , 1]                         

    men_only_stats = data[                            \

                         (data[:,4] != "female")      \

                       & (data[:,2].astype(np.float)  \

                             == i+1)                  \

                       & (data[:,9].astype(np.float)  \

                            >= j*fare_bracket_size)   \

                       & (data[:,9].astype(np.float)  \

                            < (j+1)*fare_bracket_size)\

                          , 1] 

    survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 

    survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
survival_table[ survival_table != survival_table ] = 0.5
test_file = open('../input/test.csv', 'r')

test_file_object = csv.reader(test_file)

header = next(test_file_object)

predictions_file = open("genderclassmodel.csv", "w")

p = csv.writer(predictions_file)

p.writerow(["PassengerId", "Survived"])
for row in test_file_object:                  # We are going to loop

                                              # through each passenger

                                              # in the test set                     

            

  try:                                        # Some passengers have no                                              # Fare data so try to make

    row[8] = int(float(row[8]))               # a float

    bin_fare = row[8] % fare_bracket_size

    if bin_fare >= number_of_price_brackets:

        bin_fare = number_of_price_brackets - 1

  except:                                     # If fails: no data, so 

    bin_fare = 3 - int(row[1])                # bin the fare according Pclass

  

  rnd = random.random()

  if row[3] == 'female':                      #If the passenger is female

    if rnd < survival_table[0, int(row[1])-1, bin_fare]:

        p.writerow([row[0], "1"])

    else:

        p.writerow([row[0], "0"])

  else:                                       #else if male

    if rnd < survival_table[1, int(row[1])-1, bin_fare]:

        p.writerow([row[0], "1"])

    else:

        p.writerow([row[0], "0"])

# Close out the files.

test_file.close() 

predictions_file.close()