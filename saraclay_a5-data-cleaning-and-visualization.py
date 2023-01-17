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
import pandas as pd



# Request functionalities from a bunch of different services

import pandas_datareader.data as web



# Plot library... "as" creates an alias

import matplotlib.pyplot as plt



# Helps with dates and times

import datetime as dt



# Some modules from Google that were recommended

import math

import seaborn as sns



df = Food_Establishment_Inspection_Data = pd.read_csv("../input/king-county-food-inspection/Food_Establishment_Inspection_Data.csv")
df.head(5)
four_col = df[['Program Identifier', 'Inspection Date', 'Inspection Result', 'Violation Description']]

four_col
# From above, this is the code to consolidate the table into three rows

# three_col = df[['Program Identifier', 'Inspection Result', 'Violation Description']]

# three_col



# And this is the code to identify rows with "Fremont Bowl". 

# df_filter = df['Program Identifier'].isin(['FREMONT BOWL'])

# df[df_filter]



df_filter = four_col['Program Identifier'].isin(['CAFE SELAM'])

four_col[df_filter]
four_col.tail(5)
# Let's drill this down to two columns

two_col = df[['Inspection Date', 'Inspection Result']]

two_col



# And let's see what the other results could be

two_col.head(20)

# Seems like there are only four kinds of inspection results
def inspections(results):

    if (results == "Complete"):

        return "Completed"

    elif (results == "Satisfactory"):

        return "Passed"

    elif (results == "Incomplete"):

        return "Incomplete"

    elif (results == "Unsatisfactory"):

        return "Unsatisfactory"

    else:

        return "Unknown"



df['Inspection Result'].apply(inspections).value_counts().plot(kind='bar',rot=0)
# Let's get that date converted to date-time

df['Inspection Date'] = pd.to_datetime(df['Inspection Date'])



# And this will just keep the date and get rid of the rest

# Thanks for the help, Hannah! 

df['Inspection Date'] = df['Inspection Date'].map(lambda x: x.year).fillna("")



df['Inspection Date']
df['Inspection Date'].value_counts().plot(kind='bar', figsize=(10,10), rot = 0)
df['Inspection Date'].value_counts().plot(kind='line', figsize=(10,10))
# This should do the trick...

c = df['Inspection Date'].value_counts()

print (c)
a = df['Inspection Result'].apply(inspections).value_counts()

b = df['Inspection Date'].value_counts()



df.plot(x=[a, b], kind="bar")
a = df['Inspection Result'].apply(inspections).value_counts()

c = df['Inspection Date'].value_counts()



df2 = pd.DataFrame({

    'year':['2006','2007','2008','2009','2010','2011','2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'],

    'inspections':[12245,12901,13339,13447,15583,17044,19165, 19849, 22665, 22769, 23579, 25709, 27578, 27752, 3191],

})



df2.plot(kind='bar',x='year',y='inspections',rot=0, figsize=(10,10))

plt.show()