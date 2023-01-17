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
# ../input/flights.csv



import pandas as pd



#Read the csv into a Pandas DataFrame

flights = pd.read_csv('../input/flights.csv')
#Examine the shape of the data

flights.shape
#Explore null cells

flights.isnull()
#View total of null values by column

flights.isnull().sum()
#View the number of null values in the 'TAXI_OUT' column

flights['TAXI_OUT'].isnull().sum()
#Fill all null values with a space, and score that in the current data frame

flights=flights.fillna(" ")
#Store the dataframe as a new CSV

flights.to_csv('newflights.csv', index=False)