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
#importing required packages 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#crating a Dataframe for the .csv file 

Data_frame = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

Data_frame.head()
#Getting information about the Data

#Trying to find Error or Null values in the data

Data_frame.info()
#The Review data type is Object So need to be converted to numbers

Data_frame['Reviews'] = pd.to_numeric(Data_frame['Reviews'],errors='coerce')
#Trying to find Error or Null values in the data

#only one record is null so it can be ignored

Data_frame.isnull().sum()
Data_frame.describe()#now the Rating and Review columns are Numbers
#Sorting the data in descending order with respect to Reviews column

Data_frame.sort_values('Reviews',ascending=False).head()

#it is clear the 'Facebook' has  the largest number of reviews
#displaying App which has largest number of reviews

Data_frame.sort_values('Reviews',ascending=False).iloc[0]['App']