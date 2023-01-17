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
trainset = pd.read_csv("../input/train.csv")
type(trainset)
trainset.head()
trainset.shape
trainset.tail()
trainset.describe()

#Not all ages are available, actually only 714 as displayed in count.
trainset.columns
female_psg = trainset['Sex'] == 'female'

nbr_females = np.sum(female_psg)

print("Females: ",nbr_females)

male_psg = trainset['Sex'] == 'male'

nbr_males = np.sum(male_psg)

print("Males: ",nbr_males)
female_survivors = np.sum((trainset['Sex'] == 'female') & (trainset['Survived'] == 1))

print("Female survivors: ",female_survivors)

print("Female survival rate: ", (female_survivors/nbr_females))

male_survivors = np.sum((trainset['Sex'] == 'male') & (trainset['Survived'] == 1))

print("Male survivors: ",male_survivors)

print("Male survival rate: ", (male_survivors/nbr_males))