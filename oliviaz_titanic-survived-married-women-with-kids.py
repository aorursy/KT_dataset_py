import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv



csv_file_object = csv.reader(open('../input/train.csv', 'r')) 	# Load in the csv file

next(csv_file_object) 						# Skip the fist line as it is a header

data=[] 												# Create a variable to hold the data



for row in csv_file_object: 							# Skip through each row in the csv file,

    if row[3].split(' ')[1] == "Mrs.":                  #if the woman is married(with titil "Mrs.")

        data.append([row[1],row[2],row[3],row[5],row[6],row[7]]) # adding survival,class,name,age,sibSp,parch



data = np.array(data) 									# Then convert from a list to an array.

print (np.shape(data))
survival_mask = data[0::,0] == '1' 	# mask: if the married woman survived

nonsurvival_mask = data[0::,0] == '0'  #mask: if the married woman didn't survive



kids_num_survival = data[survival_mask,5].astype(np.int)

kids_num_nonsurvival = data[nonsurvival_mask,5].astype(np.int)
kids_ratio_survival = np.sum(kids_num_survival)/np.size(kids_num_survival)

print ("survived married women have {} children in average".format(kids_ratio_survival))



kids_ratio_nonsurvival = np.sum(kids_num_nonsurvival)/np.size(kids_num_nonsurvival)

print ("nonsurvived married women have {} children in average".format(kids_ratio_nonsurvival))

no_kid_mask = data[0::,5] == '0'

no_kid_survival = data[no_kid_mask,0].astype(np.int)

print (no_kid_survival)

no_kid_survival_ratio = np.sum(no_kid_survival)/np.size(no_kid_survival)

print("{} married women with no kids survived".format(no_kid_survival_ratio))



#a married woman without children on board is very likely to survive



more_kid_mask = data[0::,5].astype(np.int) >= 3

more_kid_survival = data[more_kid_mask,0].astype(np.int)

print(more_kid_survival)

more_kid_survival_ratio = np.sum(more_kid_survival)/np.size(more_kid_survival)

print("{} married women with three or more children survived".format(more_kid_survival_ratio))



# a married woman with three or more chidlren on board is rather unlikely to survive