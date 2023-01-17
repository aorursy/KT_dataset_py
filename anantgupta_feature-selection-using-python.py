# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

people=pd.read_csv(r"../input/people.csv")

train=pd.read_csv(r"../input/act_train.csv")

print(check_output(["ls", "../input"]).decode("utf8"))



def removePrecedingChars(x):

	# We will check for the first space and then take the data from there

	#x=x.map(lambda y:y[y.index(' ')+1:50])

	if(type(x) is str):

		if(x.find(' ')==-1):

			return(x)

		else:

			return(x[x.index(' ')+1:50])

	else:

		return(x)

	

#  START OF MAIN FUNCTION



train=train[['people_id','activity_id','date','activity_category','outcome']]



# For all columns run the clean function

people=people.applymap(removePrecedingChars)

train=train.applymap(removePrecedingChars)



# We will join the datasets

fullData=pd.merge(people,train,left_on='people_id',right_on='people_id',how='inner')

del(people)

del(train)