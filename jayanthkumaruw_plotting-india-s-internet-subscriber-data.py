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
import matplotlib.pyplot as plt
#Importing the csv file into a dataframe named trai_data

trai_data = pd.read_csv("/kaggle/input/internet-user-growth-in-india-trai/Internet and teledensity_TRAI.csv")

trai_data
#Testing subsetting in Pandas with using columnn name as a method on DataFrame

trai_data.teledensity_urban
#Attempting to subset 'teledensity' from the Dataframe into a new Pandas dataframe named teledensity_data



teledensity_data = trai_data[['teledensity_rural', 'teledensity_urban'] ]

teledensity_data
#Similarly, subsetting other pertinent columns into separate dataframes

#I apologize for the inconsistent column naming scheme



broadband_data = trai_data[['Wired_Broadband_total', 'wireless_broadband_total', 'total_broadband']]

broadband_data
#Attempting to plot teledensity data

plt.plot(teledensity_data)
#Attempting to plot broadband data

plt.plot(broadband_data)
#Let's try plotting data straight from the main dataframe, subsetting whatever needed



plt.plot(trai_data[['Wired_Broadband_total', 'wireless_broadband_total', 'total_subscribers']])
