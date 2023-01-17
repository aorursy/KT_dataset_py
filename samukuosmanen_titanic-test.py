# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv', 'r')) 
header = csv_file_object.__next__()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format
        
number_passangers=np.size(data[0::,1].astype(np.float))
number_survived=np.sum(data[0::,1].astype(np.float))
proportion_survived=number_passangers/number_survived

women_only_stats=data[0::,4]=='female'
men_only_stats=data[0::,4]!='female'

women_onboard=data[women_only_stats,1].astype(np.float)
men_onboard=data[men_only_stats,1].astype(np.float)

proportion_women_survived=np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived=np.sum(men_onboard) / np.size(men_onboard)

# and then print it out
print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)



test_file = open('../input/test.csv', 'r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
prediction_file = open("genderbasedmodel.csv", "w")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()

import seaborn
