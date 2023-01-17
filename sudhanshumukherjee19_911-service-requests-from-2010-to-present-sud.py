import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
newyork = pd.read_csv('../input/311_Service_Requests_from_2010_to_Present.csv')
newyork.describe()
newyork.info()
newyork.shape
newyork.isnull().sum()
newyork.drop(['School or Citywide Complaint','Vehicle Type','Taxi Company Borough','Taxi Pick Up Location','Bridge Highway Name','Bridge Highway Direction','Road Ramp','Bridge Highway Segment','Garage Lot Name','Ferry Direction','Ferry Terminal Name'],axis=1,inplace=True)
newyork.columns
newyork.isnull().sum()
newyork.index.values
newyork.loc[:,['Complaint Type','City']]
major = newyork.loc[:,'Complaint Type']

major
major.unique()
major.nunique()
top = major.value_counts()

top
top.head(10)
major.value_counts().plot(kind='bar',title='Count vs Complaint types',color= 'red')
top.plot(kind='hist',title='Vizualize the complaint types',color='red')
top.head(10).plot(kind='bar', title = 'Visualize the Complaint Types',color='red')