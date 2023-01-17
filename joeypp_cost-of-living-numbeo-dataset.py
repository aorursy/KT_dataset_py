# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        inp=os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
inpfile=pd.read_csv(inp)



#Subset the data with rows having attributes of interest

inp_subset=inpfile[inpfile[inpfile.columns.values[0]].isin([

    'Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate',

    'Price per Square Meter to Buy Apartment in City Centre',

    'Average Monthly Net Salary (After Tax)','Gasoline (1 liter)', 

    'Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)',

    'International Primary School, Yearly for 1 Child',

    'Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment'])].reset_index().drop(columns=['index'])



inp_subset.columns.values[0]='Attribute'



#Convert Attribute rows into columns.

inp_subset=inp_subset.melt(id_vars='Attribute', var_name=['City'], value_name='Cost in EUR')



#Convert the entire dataframe using pivot, so we will have 'Country' column and a separate column for each attribute (cost of living factor)

inp_subset=inp_subset.pivot_table(index=['City'],columns='Attribute',values='Cost in EUR',aggfunc='first').reset_index()

inp_subset=inp_subset[[

    'City',

    'Average Monthly Net Salary (After Tax)',

    'Gasoline (1 liter)',

    'International Primary School, Yearly for 1 Child',

    'Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)',

    'Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate',

    'Price per Square Meter to Buy Apartment in City Centre',

    'Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment']]



inp_subset.columns=['City', 'Monthly Salary', 'Gasoline','Yearly School Fee', 'Internet', 'Mortgage %', 'House Price per sqm','Basic Amenities']



#Lets create 'Country' based on 'City' field

inp_subset['Country']=inp_subset['City'].apply(lambda x: x.split(',')[-1].strip(' '))



#Remove Country from City field

inp_subset['City']=inp_subset['City'].apply(lambda x: x.split(',')[0].strip(' '))



inp_subset['School Fee (% of Salary)']=inp_subset['Yearly School Fee']*100/(inp_subset['Monthly Salary']*12)

inp_subset['Cost of 100 sqm house']=inp_subset['House Price per sqm']*100

inp_subset['Annual house payment']=((inp_subset['Cost of 100 sqm house']*inp_subset['Mortgage %']/100)+inp_subset['Cost of 100 sqm house'])/20

inp_subset['Housing Debt to income ratio']=inp_subset['Annual house payment']*100/(inp_subset['Monthly Salary']*12)

inp_subset['Basic Amenities']=inp_subset['Basic Amenities']*100/inp_subset['Monthly Salary']

inp_subset.head()
sns.pairplot(inp_subset[['Monthly Salary', 'Yearly School Fee', 'Mortgage %', 'House Price per sqm','Basic Amenities','Housing Debt to income ratio']])
for columns in ['Monthly Salary', 'Mortgage %', 'House Price per sqm','Basic Amenities', 'Yearly School Fee','Housing Debt to income ratio']:

    plt.figure()

    sns.boxplot(y = columns, data = inp_subset)
inp_indiaandus=inp_subset[(inp_subset['Country'] == 'India') | (inp_subset['Country'] == 'United States')]



for columns in ['Monthly Salary', 'Mortgage %', 'House Price per sqm','Basic Amenities', 'Yearly School Fee','Housing Debt to income ratio']:

    plt.figure()

    sns.boxplot(x='Country' , y=columns, data = inp_indiaandus)

inp_india = inp_indiaandus[inp_indiaandus['Country'] == 'India']

inp_us = inp_indiaandus[inp_indiaandus['Country'] == 'United States']



#India

for columns in ['Monthly Salary', 'Mortgage %', 'House Price per sqm','Basic Amenities', 'Yearly School Fee','Housing Debt to income ratio']:

    plt.figure(figsize=(20,2))

    sns.scatterplot(x='City' , y=columns, data = inp_india)
#United States

for columns in ['Monthly Salary', 'Mortgage %', 'House Price per sqm','Basic Amenities', 'Yearly School Fee','Housing Debt to income ratio']:

    plt.figure(figsize=(20,2))

    sns.scatterplot(x='City' , y=columns, data = inp_us)
#Load dataset and filter data for the cities I am interested in.

inp_cities=pd.read_csv(inp)

inp_cities.columns.values[0]='Attribute'



#Convert Attribute rows into columns.

inp_cities=inp_cities.melt(id_vars='Attribute', var_name=['City'], value_name='Cost in EUR')



#Convert the entire dataframe using pivot, so we will have 'Country' column and a separate column for each attribute (cost of living factor)

inp_cities=inp_cities.pivot_table(index=['City'],columns='Attribute',values='Cost in EUR',aggfunc='first').reset_index()



#Lets create 'Country' based on 'City' field

inp_cities['Country']=inp_cities['City'].apply(lambda x: x.split(',')[-1].strip(' '))



#Remove Country from City field

inp_cities['City']=inp_cities['City'].apply(lambda x: x.split(',')[0].strip(' '))



inp_cities['School Fee (% of Salary)']=inp_cities['International Primary School, Yearly for 1 Child']*100/(inp_cities['Average Monthly Net Salary (After Tax)']*12)

inp_cities['Cost of 100 sqm house']=inp_cities['Price per Square Meter to Buy Apartment in City Centre']*100

inp_cities['Annual house payment']=((inp_cities['Cost of 100 sqm house']*inp_cities['Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate']/100)+inp_cities['Cost of 100 sqm house'])/20

inp_cities['Housing Debt to income ratio']=inp_cities['Annual house payment']*100/(inp_cities['Average Monthly Net Salary (After Tax)']*12)

inp_cities['Basic Amenities']=inp_cities['Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment']*100/inp_cities['Average Monthly Net Salary (After Tax)']



inp_cities=inp_cities[(inp_cities['City'] == 'Dallas') | (inp_cities['City'] == 'Chicago') | (inp_cities['City'] == 'Seattle') | (inp_cities['City'] == 'San Diego')]
inp_cities