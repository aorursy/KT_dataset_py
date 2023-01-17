import numpy as np

import csv as csv



csv_file_object = csv.reader(open("../input/train.csv" , "rt", newline=''))

header = next(csv_file_object)

data = []



for row in csv_file_object:

    data.append(row[0:])

data = np.array(data)



# to find the proportion that survived we declare some variables.

number_passengers = np.size(data[:,1].astype(np.float))

number_survived = np.sum(data[:,1].astype(np.float))

proportion_survivors = number_survived/number_passengers



# differentiating the men/women onboard. Identifies the rows with the women.

women_stats = data[:,4] == 'female'

men_stats = data[:,4] != 'female'



# masking.

women_onboard = data[women_stats,1].astype(np.float)

men_onboard = data[men_stats,1].astype(np.float)



proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)

proportion_women_survived = np.sum(women_onboard)/np.size(women_onboard)



print("Proportion of women survived is: " + str(proportion_women_survived))

print('Proportion of men survived is: ' + str(proportion_men_survived))



# predicting that if women survived and if man not survived.



test_file = open('../input/test.csv',"rt", newline='')

test_file_object = csv.reader(test_file)

header = next(test_file_object )



      

predictions_file = open('gendermodel.csv' , 'wt',newline='')      

predictions_file_object = csv.writer(predictions_file)

# give column headers to these files

predictions_file_object.writerow(['PassengerId', 'Survived'])



# conditional loop based on the model

for row in test_file_object:

      if row[3] == 'female':

          predictions_file_object.writerow([row[0],'1'])

      else:

          predictions_file_object.writerow([row[0],'0'])

 

test_file.close()

predictions_file.close() 
import numpy as np

import csv as csv



csv_file_object = csv.reader(open("../input/train.csv" , "rt", newline=''))

header = next(csv_file_object)

data = []



for row in csv_file_object:

    data.append(row[0:])

data = np.array(data)



fare_ceiling = 40

# putting the data below the ceiling

data[(data[:,9]).astype(np.float) >= 40 , 9] = fare_ceiling - 1

fare_bracket_size = 10

number_of_brackets = fare_ceiling/fare_bracket_size 

number_of_classes = len(np.unique(data[:,2]))



# now the idea is to create a 2*3*4 matrix based on the different values of gender, class and fare bin 

# so that it can be used as a reference

survival_table = np.zeros([2,number_of_classes,number_of_brackets],float)



for i in range(number_of_classes):

    for j in range(int(number_of_brackets)):

        women_only_stats = data[(data[:,4] =='female')\

                                & (data[:,2].astype(np.int) == i + 1)\

                                & ((data[:,9]).astype(np.float) >= fare_bracket_size*j)\

                                & (data[:,9].astype(np.float) < fare_bracket_size*(j+1)),1]

        men_only_stats = data[(data[:,4] == 'male')\

                              & (data[:,2] == i + 1)\

                              & (data[:,9].astype(np.float) >= fare_bracket_size*j)\

                              & (data[:,9].astype(np.float) < fare_bracket_size*(j+1)),1]

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))

        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

        

survival_table[survival_table != survival_table] = 0



survival_table[ survival_table < 0.5 ] = 0

survival_table[ survival_table >= 0.5 ] = 1



# working with the test file

test_file = open('../input/test.csv',"rt", newline='')

test_file_object = csv.reader(test_file)

header = next(test_file_object )



predictions_file = open('genderclass.csv' , 'wt',newline='')      

predictions_file_object = csv.writer(predictions_file)

# give column headers to these files

predictions_file_object.writerow(['PassengerId', 'Survived'])



# Making the fare data in the test file binned.



for row in test_file_object:

    for j in range(int(number_of_brackets)):

        try:

            row[8] = float(row[8])

        except:

            bin_fare = 3 - float(row[1])

            break

        if row[8] > fare_ceiling:

            bin_fare = number_of_brackets - 1

            break

        if row[8] >= j*fare_bracket_size and row [8] < (j+1)*fare_bracket_size:

            bin_fare = j

            break

            

    if row[3] == 'female':

         predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])

    else:

        predictions_file_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])



# Close out the files

test_file.close()

predictions_file.close()

        