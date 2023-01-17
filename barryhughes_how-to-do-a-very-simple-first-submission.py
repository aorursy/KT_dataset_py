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
# import files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# I have imported the training file but as we know what we want to submit already we won't need

# to use it to train a prediction model. We will just add what our females survive, males don't

# prediction to the test file and the submit it.



#preview the test file just to see it

test.head()
# this is the first step which creates Subfile which contains just the two columns we want.

Subfile = test.loc[:, ['PassengerId', 'Sex']]

# we will then have a look at the first few rows to check it looks ok

Subfile.head()
# we now need to change male to 0 and female to 1 in the Sex column



Subfile['Sex'] = Subfile['Sex'].replace(['male'],0)

Subfile['Sex'] = Subfile['Sex'].replace(['female'],1)



# and let's look at the first few rows to double check

Subfile.head()
# now we just need to rename our 'Sex' column as 'Survived'



Subfile.columns = ['PassengerID', 'Survived']



#and one final look...

Subfile.head()
# We now have the simple submission data that we want so we need to create a .csv file from it



Subfile.to_csv('titanic.csv', index=False)



#the index=False bit just takes off the index column so we are left with just the two we want
