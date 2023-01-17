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
#This is a dataset of regional agricultural groups. 

#I wanted to know what the demographics were of people who owned agricultural operations and stumbled upon this data set

#def var for dataframe

reg_agricultureData = pd.read_csv("/kaggle/input/REG_profile_data.csv")
#using pandas and matplotlib

import pandas_datareader.data as web

import matplotlib.pyplot as plt

#checking to see what dataframe looks like

reg_agricultureData
#plotting County FIPS column only 

reg_agricultureData['County FIPS'].plot()

def value_rank(value_c):

    if value_c <-1000: 

        return "Poor"

    elif value_c > 80000:

        return 'Average'

    elif stock_price > 2000000:

        return 'Stellar'
#not sure why this isn't making a bar graph of the value column

reg_agricultureData['Value'].apply(value_rank).value_count().plot(kind='barh')
reg_agricultureData['State FIPS'].plot()
#trying to manipulate data to show groups as columns and which states the groups are located in

reg_agricultureData.pivot(columns="Group", values="State")
#def var for the new dataframe

group_df = reg_agricultureData.pivot(columns="Group", values="State")

#group_df['Farms with women principal operators'].count() counts the numbers in a specific column

#printing total in the "..women principal.."column

print("Number of Farms with women principal operators" +':'+str(group_df['Farms with women principal operators'].count()))

#group_df['Farms with women principal operators'].count() 