""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this

Author : AstroDave

Date : 18 September 2012

Revised: 28 March 2014



"""





import csv as csv

import numpy as np



csv_file_object = csv.reader(open('../input/train.csv', 'r')) 	# Load in the csv file

header = next(csv_file_object) 						# Skip the fist line as it is a header

data=[] 												# Create a variable to hold the data



for row in csv_file_object: 							# Skip through each row in the csv file,

    data.append(row[0:]) 								# adding each row to the data variable

data = np.array(data) 									# Then convert from a list to an array.



# Now I have an array of 12 columns and 891 rows

# I can access any element I want, so the entire first column would

# be data[0::,0].astype(np.float) -- This means all of the rows (from start to end), in column 0

# I have to add the .astype() command, because

# when appending the rows, python thought it was a string - so needed to convert



# Set some variables

number_passengers = np.size(data[0::,1].astype(np.float))

number_survived = np.sum(data[0::,1].astype(np.float))

proportion_survivors = number_survived / number_passengers 



print("Number of people: %s" % number_passengers)

print("Number of people who survived: %s" % number_survived)



# I can now find the stats of all the women on board,

# by making an array that lists True/False whether each row is female

women_only_stats = data[0::,4] == "female" 	# This finds where all the women are

men_only_stats = data[0::,4] != "female" 	# This finds where all the men are (note != means 'not equal')



# I can now filter the whole data, to find statistics for just women, by just placing

# women_only_stats as a "mask" on my full data -- Use it in place of the '0::' part of the array index. 

# You can test it by placing it there, and requesting column index [4], and the output should all read 'female'

# e.g. try typing this:   data[women_only_stats,4]

women_onboard = data[women_only_stats,1].astype(np.float)

men_onboard = data[men_only_stats,1].astype(np.float)



# and derive some statistics about them

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)



print("Number of women: %s" % np.size(women_onboard))

print("Number of women who survived: %s" % np.sum(women_onboard))

print("Number of men: %s" % np.size(men_onboard))

print("Number of men who survived: %s" % np.sum(men_onboard))

print ('Proportion of women who survived is %s' % proportion_women_survived)

print ('Proportion of men who survived is %s' % proportion_men_survived)
samp = np.array(data[0::,2])

np.unique(samp)
class_only_stats = []

class_onboard = []

proportion_class_survived = []

number_text = ["one", "two", "three"]



for i in range(1,4):

    class_only_stats.append(data[0::,2] == str(i))

    class_onboard.append(data[class_only_stats[i-1],1].astype(np.float))

    proportion_class_survived.append(np.sum(class_onboard[i-1]) / np.size(class_onboard[i-1]))

    

    print('Number of class,', number_text[i-1], 'people: %s' %np.size(class_onboard[i-1]))

    print('Number of class,', number_text[i-1], 'people who survived: %s' %np.sum(class_onboard[i-1]))

    

for i in range(0,3):

    print('Proportion of class', number_text[i], 'who survived is %s' % proportion_class_survived[i])
samp = np.array(data[0::,6])

np.unique(samp)
sibSpo_only_stats = []

sibSpo_onboard = []

proportion_sibSpo_survived = []



#print('Proportion of people with n siblings/spouses:')

for i in range(0, 7):

    x = -1

    if i == 6:

        x = 8

    else: x = i

        

    sibSpo_only_stats.append(data[0::,6] == str(x))

    sibSpo_onboard.append(data[sibSpo_only_stats[i],1].astype(np.float))

    print("n = ",x)

    print("total: %s"% np.size(sibSpo_onboard[i]))

    print("survived: %s"% np.sum(sibSpo_onboard[i]))

    proportion_sibSpo_survived.append(np.sum(sibSpo_onboard[i]) / np.size(sibSpo_onboard[i]))

    print ('Proportion: %s' % proportion_sibSpo_survived[i])

    print("\n")
samp = np.array(data[0::,7])

np.unique(samp)
Parch_only_stats = []

Parch_onboard = []

proportion_Parch_survived = []



#print('Proportion of people with n siblings/spouses:')

for i in range(0, 7):

    x = i

        

    Parch_only_stats.append(data[0::,7] == str(x))

    Parch_onboard.append(data[Parch_only_stats[i],1].astype(np.float))

    print("n = ",x)

    print("total: %s"% np.size(Parch_onboard[i]))

    print("survived: %s"% np.sum(Parch_onboard[i]))

    proportion_Parch_survived.append(np.sum(Parch_onboard[i]) / np.size(Parch_onboard[i]))

    print ('Proportion: %s' % proportion_Parch_survived[i])

    print("\n")
C_only_stats = data[0::,11] == "C" 	

Q_only_stats = data[0::,11] == "Q" 

S_only_stats = data[0::,11] == "S" 



C_onboard = data[C_only_stats,1].astype(np.float)

Q_onboard = data[Q_only_stats,1].astype(np.float)

S_onboard = data[S_only_stats,1].astype(np.float)



proportion_C_survived = np.sum(C_onboard) / np.size(C_onboard)

proportion_Q_survived = np.sum(Q_onboard) / np.size(Q_onboard)

proportion_S_survived = np.sum(S_onboard) / np.size(S_onboard)



print('Number of people who embarked on C: %s'%np.sum(C_onboard))

print('Number of people who embarked on C that survived: %s'%np.size(C_onboard))

print('Number of people who embarked on Q: %s'%np.sum(Q_onboard))

print('Number of people who embarked on Q that survived: %s'%np.size(Q_onboard))

print('Number of people who embarked on S: %s'%np.sum(S_onboard))

print('Number of people who embarked on S that survived: %s'%np.size(S_onboard))



print ('Proportion of who embarked on Cherbourg who survived is %s' % proportion_C_survived)

print ('Proportion of who embarked on Queenstown who survived is %s' % proportion_Q_survived)

print ('Proportion of who embarked on Southampton who survived is %s' % proportion_S_survived)
import pandas as pd

titanicData = pd.read_csv('../input/train.csv')

print(titanicData)
def string_to_int_gender(value):

    try:

        if value == "male":

            return 1;

        else:

            return 2;

    except ValueError:    

        return None

    

titanicData["Sex"] = titanicData["Sex"].apply(string_to_int_gender)
def string_to_int_embarkation(value):

    try:

        if value == "C":

            return 1;

        elif value == "Q":

            return 2;

        else:

            return 3

    except ValueError:    

        return None

    

titanicData["Embarked"] = titanicData["Embarked"].apply(string_to_int_embarkation)

def reverse_class(value):

    try:

        if value == 1:

            return int(3);

        elif value == 3:

            return int(1);

        elif value == 2:

            return int(2);

    except ValueError:    

        return None

    

titanicData["Pclass"] = titanicData["Pclass"].apply(reverse_class)

titanicData
print(titanicData.dtypes)

titanicData[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].corr()
import pandas_profiling

pandas_profiling.ProfileReport(titanicData)
for possible_value in set(titanicData["Cabin"].tolist()): #a set maintains only unique values

    print(possible_value)