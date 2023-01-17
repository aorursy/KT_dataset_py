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
#load the csv file using pd.read_csv method and save into a variable flavors that is also called a dataframe

flavors  = pd.read_csv("../input/flavors_of_cacao.csv")



''' for printing  the begining five and as we want then we use dataframe.head()

or for last lines use tail() method'''

flavors.head()
#to know columns names

flavors.columns
# use info method to know about null entries 

flavors.info()

#here we find two columns which have null entries

#to know how much column entries have columns 

flavors['Bean\nType'].isnull().sum()

#this colummn has only one null entry

#now check for other column'''

flavors['Broad Bean\nOrigin'].isnull().sum()
#fill up null entries by fillna function

flavors.fillna(method='bfill',inplace=True)

# use method to fill null entries by backward propogation and inplace =True is  used to modify the result permanent

# again check for null entries have or  not

flavors.isnull().sum()

#wow! no any columns have any null entries so we  move on next steps
#print the row which have maximum rating

#flavors.loc[flavors['Rating'].idxmax()]

flavors[flavors['Rating'] ==flavors['Rating'].max()]

#two rows printed which have maximum rating 5.0

# conclusion - only company Amedei make the choklate which are so much popular and 

#and the choklate have 70% Cocoa percent
#check the rating of choklate which have max cocoa percentage

#flavors.groupby('Rating')['Cocoa\nPercent'].count()

ratmax=flavors[flavors['Cocoa\nPercent']=='100%']

ratmax['Rating']

#only twenty entries which have cocoa-percenatge =100% and their rating lies bw 1.00 and 3.00

#conclusion2- the choklates which have cocoa-percentage above 70% are not so much liked by peoples

#and the choklates which have coaco percenatge 50_70% are so much liked by people
# visualization b/w rating and review_date

import matplotlib.pyplot as plt

plt.scatter(flavors['Review\nDate'],flavors['Rating'])
