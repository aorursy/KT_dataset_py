import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/master.csv')

data.head() 
# This code line get all columns of into dataset

data.columns 
# It shows that this csv document occurs 27820 items and 12 columns

data.shape
# info gives data type like dataframe

data.info()
print(data['country'].value_counts(dropna = False))
data.describe()
#Melt 

melted = pd.melt(frame=data,id_vars = 'country', value_vars = ['population','age'])

melted
#Concatenating Data and Data Types

data['population'] = data['population'].astype('float')

data.info()
data.rename(columns = {'HDI for year':'HDI_for_year'},inplace=True)

data["HDI_for_year"].value_counts(dropna = False)
data["HDI_for_year"].dropna(inplace=True)

assert data["HDI_for_year"].notnull().all()