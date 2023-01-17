# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


while True: 





        if year % 4 == 0 and year % 100 == 0 and year % 400 == 0 : #if statement is true for year entered by user then below will be printed

            print(str(year) + ' is a leap year') #if the year fullfills all the requirements listed above than this statement will print

        elif year %  4 != 0 : #if statement is true for year entered by user then below will be printed

            print(str(year) + ' is not a leap year') #if the year fullfills all the requirements listed above than this statement will print

        elif year % 100 != 0: #if statement is true for year entered by user then below will be printed

            print(str(year) + ' is a leap year') #if the year fullfills all the requirements listed above than this statement will print

        elif year % 400 != 0: #if statement is true for year entered by user then below will be printed

            print(str(year) + ' is not a leap year') #if the year fullfills all the requirements listed above than this statement will print

        else: #else statement that runs if user passes none of the requirments above

            print('Error') #This will print when nothing works
import pandas as pd

expense = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv")
#To view the data set 



#replacing the Null values with zero

expense.replace(np.nan, 0, inplace=True)

#view table

#get specific columns of a table





#get specific rows of a table

print (expense.iloc[0:20, 4:63])
#getting some form of data

#the maximum expense

print (expense.iloc[:,5:63].max())







#getting more specific data

expense.set_index("Name", inplace=True)

expense.head()

expense.loc[['Zambia'],['Code','Type','2018']]