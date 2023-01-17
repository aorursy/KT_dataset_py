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
#Reads the dataset "Netflix Shows", encoding = 'latin1' was passed because possibly, there are characters

#that need to be escaped.

netflixData = pd.read_csv("../input/Netflix Shows.csv", encoding='latin1')

#Displays the first five entries in the dataset

netflixData.head()
#Checks if the dataset is cleaned and/ or complete by using the count method. 

#This tells us if there are null or NaN values in the indices

netflixData.count()
#What we can do to "clean" the data is to delete the rows where NaN values exists. To do so, we set

#another dataFrame object based from the original dataFrame with only the not null values.

newDataset=netflixData[netflixData.ratingLevel.notnull()]



#We can verify that the entries with null values were removed using the same count method

newDataset.count()
newDataset['user rating score'].fillna(0, inplace=True)

newDataset.count()
average=newDataset['user rating score'].fillna(newDataset['user rating score'].mean())

average.count()
zeros.mean()