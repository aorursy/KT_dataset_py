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
import numpy as np # linear algebra

import csv as csv # data processing, CSV file I/O (e.g. pd.read_csv)



#Print you can execute arbitrary python code

train = csv.reader(open("../input/train.csv", 'r') )



#Print to standard output, and see the results in the "log" section below after running your script

print("\n\nTop of the training data:")

header=next(train)

data=[]

for row in train:

    data.append(row)

data=np.array(data)

print(data[-1])

no_passenger=np.size(data[0::,1].astype(np.float))

print(no_passenger)

no_survived=np.sum(data[0::,1].astype(np.float))

print(no_survived)

proportion_of_survivor=no_survived/no_passenger

print(proportion_of_survivor)

women_only=data[0::,4]=='female'

men_only=data[0::,4]!='male'

#print(women_only)

women_on_board=data[women_only,1].astype(float)

#print(women_on_board)

men_on_board=data[men_only,1].astype(float)

proportion_men_survived=np.sum(men_on_board)/np.size(men_on_board)

print(proportion_men_survived)

proportion_women_survived=np.sum(women_on_board)/np.size(women_on_board)

print(proportion_women_survived)

test1=open('../input/test.csv','r')

test=csv.reader(test1)

header=next(test)

prediction_file1=open("gendermodel.csv","w")

prediction_file=csv.writer(prediction_file1)

prediction_file.writerow(["passenger_id","survived"])

for row in test:

     if row[3]=='female':

          prediction_file.writerow([row[0],'1'])

     else:

          prediction_file.writerow([row[0],'0'])

test1.close()

prediction_file1.close()









    