# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# lets discover the files that we will work

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# output

# deputies_dataset.csv

# dirty_deputies_v2.csv



#bring in the six packs

# lets bring the csv files to a dataframe

depData = pd.read_csv('../input/deputies_dataset.csv',delimiter=',')

dirtDepData = pd.read_csv('../input/dirty_deputies_v2.csv',delimiter=',')



# lets explore the data set

# first we see the size of the two datasets

print('Deputies Dataset size (rows,columns) = ',depData.shape)

print('Dirty Deputies Dataset size (rows,columns) = ',dirtDepData.shape)



# other way to do it

print('Deputies Dataset rows = ',depData.shape[0],' columns = ',depData.shape[1])

print('Dirty Deputies Dataset rows = ',dirtDepData.shape[0],'columns = ',dirtDepData.shape[1])



# and another way to do

print('Deputies Dataset rows = ',len(depData.axes[0]),' columns = ',len(depData.axes[1]))

print('Dirty Deputies Dataset rows = ',len(dirtDepData.axes[0]),'columns = ',len(dirtDepData.axes[1]))



# now that we know the size we can explore the data

# lets sumarize the datas

print(depData.describe())

print(dirtDepData.describe())



# lets see if we have null data , where and how many

print(depData.isnull().sum())

print(dirtDepData.isnull().sum())





# what type of data do we have?

print(depData.dtypes)

print(dirtDepData.dtypes)





#lets separete the data in two kinds numerical and categorical

numerical_depData = pd.DataFrame.select_dtypes(depData,include=[np.number])

print(numerical_depData.dtypes)

categorical_depData = pd.DataFrame.select_dtypes(depData,include=[np.object])

print(categorical_depData.dtypes)



numerical_dirtDepData = pd.DataFrame.select_dtypes(dirtDepData,include=[np.number])

print(numerical_dirtDepData.dtypes)

categorical_dirtDepData = pd.DataFrame.select_dtypes(dirtDepData,include=[np.object])

print(categorical_dirtDepData.dtypes)









# Any results you write to the current directory are saved as output.