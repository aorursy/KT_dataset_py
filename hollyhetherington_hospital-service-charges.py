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
#INTRODUCTION

#I found this dataset interesting because until Jan of last year, hospitals weren't required to publish charges for their services. 

#As I've recently lost access to good health insurance, and according to a Census Bureau report from 2019, an estimated 27.5 million people, 8.5% of the population, went without health insurance in 2018,

#I'm interested in what it would actually set me and others back if for some reason I had to go to the hospital. 



#Use the Pandas techniques we've learned to prepare and shape your data. If the data is already cleaned then you should decide what parts of the data are most interesting to you and will be useful to visualize.

#Use some simple visualization techniques to represent your data. You should aim to produce at least two or three tables and charts of various, suitable types (line, bar, pie)

#Make sure to document each step of the way with a Text Cell

#You'll have time after the class session to put the finishing touches on your notebook.
import pandas as pd

charges = pd.read_csv("../input/inpatient-hospital-bills/Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_the_Top_100_Diagnosis-Related_Groups__DRG__-_FY2011.csv")
#Review the first few rows of the dataset

charges.head()
#Determine how many columns are in the dataset

len(charges.columns)
#Determine which columns are available in the dataset



charges.dtypes
#Create a dataframe of select columns

#df1 = df[['a','b']]



charges_small = charges[['DRG Definition','Provider State','Average Medicare Payments']]
charges_small.head()
#Importing matplotlib



import matplotlib.pyplot as plt

#create a scatterplot of treatmetns and medicare payments 

#This did not work



x = charges_small[['DRG Definition']]

y = charges_small[['Average Medicare Payments']]

size=charges_small[['Displacement']]

charges_small.plot(kind='scatter', x='DRG Definition', y='Average Medicare Payments', s=size, alpha=.5)

#S = size and alpha = brightness/opacity 
#Plot two lines on a graph

#df.plot(y="High")



charges_small.plot(y="Average Medicare Payments", x="DRG Definition")



#Attempt to plot a bar chart



payment=charges_small



def payment_buckets(average_payment):

    if average_payment <100: 

        return"Low"

    elif stock_price >100 and stock_price <=1000:

        return "Middle"

    elif stock_price >1200:

        return "Yikes"
payment['Results'].apply(charges_small)
payment['Results'].apply(charges_small).value_counts().plot(kind='bar')