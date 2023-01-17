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
import csv as csv

import numpy as np





#READ TRAINING DATASET

csv_database_obj=csv.reader(open("../input/train.csv",'r'))

header=csv_database_obj.__next__()      #Skips the first line

data=[]                                 #Create data file



for row in csv_database_obj:            #Go through all columns of the csv database object and append each row in the file data

    data.append(row)



data=np.array(data)                     #Convert the list data to array. Note each element is String in this format.

print(data[0])





number_of_passengers=np.size(data[0::,1].astype(np.float))

number_survived=np.sum(data[0::,1].astype(np.float))



women_only_studies = data[0::,4] == "female"                      #Get boolean vector specifying indexs where we have women

men_only_studies = data[0::,4] != "female"



women_on_board = data[women_only_studies,1].astype(np.float)      #Get a vector of women on board with 0 if they didn't survive and 1 otherwise

men_on_board = data[men_only_studies,1].astype(np.float)



num_survived_women = np.sum(women_on_board)

num_onboard_women = np.size(women_on_board)

num_survived_men = np.sum(men_on_board)

num_onboard_men = np.size(men_on_board)



proportion_women_alive = num_survived_women/num_onboard_women               #Compute proportion of women alive

proportion_men_alive = num_survived_men/num_onboard_men

print('The proportion of women alive after Titanic crash is %s' % proportion_women_alive)

print('The proportion of men alive after Titanic crash is %s' % proportion_men_alive)







#READ TEST DATASET

test_file = open("../input/test.csv", 'r')

test_file_object = csv.reader(test_file)

header = test_file_object.__next__()





#PYTHONISING THE SECOND SUBMISION

# So we add a ceiling. 

fare_ceiling = 40

data[data[0::,9].astype(np.float) >= fare_ceiling, 9]= fare_ceiling-1.0 #Anything bigger than 40 goes to 39



fare_bracket_size = 10

number_of_price_brackets = fare_ceiling / fare_bracket_size



# Calculate the number of classes on board

# Take the length of an array of unique values in column index 2

number_of_classes = len(np.unique(data[0::,2])) 



# Initialize the survival table with all zeros

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))





for i in range(number_of_classes):       #loop through each class

  for j in range(int(number_of_price_brackets)):   #loop through each price bin



    women_only_stats = data[                          #Which element           

                         (data[0::,4] == "female")    #is a female

                       &(data[0::,2].astype(np.float) #and was ith class

                             == i+1)                        

                       &(data[0:,9].astype(np.float)  #was greater 

                            >= j*fare_bracket_size)   #than this bin              

                       &(data[0:,9].astype(np.float)  #and less than

                            < (j+1)*fare_bracket_size)#the next bin    

                          , 1]                        #in the 2nd col                           

 						                                    									





    men_only_stats = data[                            #Which element           

                         (data[0::,4] != "female")    #is a male

                       &(data[0::,2].astype(np.float) #and was ith class

                             == i+1)                                       

                       &(data[0:,9].astype(np.float)  #was greater 

                            >= j*fare_bracket_size)   #than this bin              

                       &(data[0:,9].astype(np.float)  #and less than

                            < (j+1)*fare_bracket_size)#the next bin    

                          , 1] 

                                                 

    survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 

    survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

    

survival_table[ survival_table != survival_table ] = 0.



survival_table[ survival_table < 0.5 ] = 0

survival_table[ survival_table >= 0.5 ] = 1
