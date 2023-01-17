import csv as csv
import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.

from subprocess import check_output
print("List of files in ../input directory:\n")
print(check_output(["ls", "../input"]).decode("utf8"))

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv', "rt", newline='\n')) 
header = next(csv_file_object)   # The next() command just skips the 
                                 # first line which is a header
data=[]
for row in csv_file_object:
    data.append(row)
data = np.array(data)

print("Header:")
print(header)
print("\nFirst row of data: ")
print(data[0])
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print("Number of passengers: {0}".format(number_passengers))
print("Number of survivors: {0}".format(number_survived))
print("Proportion: {0:.2f}%".format(proportion_survivors * 100))
women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

print("Number of women: {0}".format(np.sum(women_onboard).astype('int')))
print("Number of men: {0}".format(np.sum(men_onboard).astype('int')))
print()

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard) * 100
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) * 100

print('Proportion of women who survived is {0:.2f}%'.format(proportion_women_survived))
print('Proportion of men who survived is {0:.2f}%'.format(proportion_men_survived))

test_file = open('../input/test.csv', 'rt', newline="\n")
test_file_object = csv.reader(test_file)
header = next(test_file_object)

prediction_file = open("genderbasedmodel.csv", "wt", newline="\n")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
    if row[3] == 'female':
        prediction_file_object.writerow([row[0],'1'])

    else:
        prediction_file_object.writerow([row[0],'0'])
        
test_file.close()
prediction_file.close()
np.set_printoptions(threshold=np.nan)

# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
# but this is unnecessary for training data
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

print("Number of price brackets: {0}".format(number_of_price_brackets))

# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::,2]))

print("Number of classes: {0}".format(number_of_classes))

# Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))
for i in range(number_of_classes):       #loop through each class
    for j in range(number_of_price_brackets):   #loop through each price bin
        women_only_stats = data[
                         (data[0::, 4] == "female")
                       & (data[0::, 2].astype(np.float) == i + 1)
                       & (data[0:, 9].astype(np.float) >= j * fare_bracket_size)
                       & (data[0:,9].astype(np.float) < (j + 1) * fare_bracket_size), 1]

        men_only_stats = data[
                         (data[0::,4] != "female")
                       & (data[0::,2].astype(np.float) == i+1)
                       & (data[0:,9].astype(np.float) >= j * fare_bracket_size)
                       & (data[0:,9].astype(np.float) < (j + 1) *fare_bracket_size), 1]
        
        survival_table[ survival_table != survival_table ] = 0.
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

print("Survival table (sex, class, fare):")
print(survival_table)

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

print()
print("Survival table (sex, class, fare):")
print(survival_table)
test_file = open('../input/test.csv', 'rt')
test_file_object = csv.reader(test_file)
header = next(test_file_object)

predictions_file = open("genderclassmodel.csv", "wt")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:                          # We are going to loop
                                                      # through each passenger
                                                      # in the test set                     
    for j in range(number_of_price_brackets):         # For each passenger we
                                                      # loop thro each price bin
        try:                                          # Some passengers have no
                                                      # Fare data so try to make
            row[8] = float(row[8])                    # a float
        except:                                       # If fails: no data, so 
            bin_fare = 3 - float(row[1])              # bin the fare according Pclass
            break                                     # Break from the loop
        if row[8] > fare_ceiling:                     # If there is data see if
                                                      # it is greater than fare
                                                      # ceiling we set earlier
            bin_fare = number_of_price_brackets - 1   # If so set to highest bin
            break                                     # And then break loop
        if row[8] >= j * fare_bracket_size and row[8] < (j + 1) * fare_bracket_size:   # If passed these tests 
                                                      # then loop through each bin 
            bin_fare = j                              # then assign index
            break

    if row[3] == 'female':                             #If the passenger is female
        p.writerow([row[0], "%d" % int(survival_table[0, int(float(row[1]) - 1), int(bin_fare)])])
    else:                                          #else if male
        p.writerow([row[0], "%d" % int(survival_table[1, int(float(row[1]) - 1), int(bin_fare)])])
     

test_file.close() 
predictions_file.close()

# print(check_output(["cat", "genderclassmodel.csv"]).decode("utf8"))
