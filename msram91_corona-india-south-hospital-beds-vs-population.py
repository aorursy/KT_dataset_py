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
import seaborn as sns

import matplotlib.pyplot as plt
df_Hospital =  pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")

df_Hospital.head()
df_population = pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")

df_population.head()

# Check the info of the Hospital Dataset

df_Hospital.info()
df_Hospital.shape   #38 rows and 12 columns
# We have to convert datatypes of some object columns into Numeric.It is required to do data analysis

df_Hospital_cols_obj=df_Hospital.select_dtypes(include = ['object']).columns.to_list()

df_Hospital_cols_obj=df_Hospital_cols_obj[1:]   # removing state columns



# apply datatype function



for i in df_Hospital_cols_obj:

    df_Hospital[i]=df_Hospital[i].str.replace(',', '').apply(pd.to_numeric)



# Check the datatype

df_Hospital.info()



# Null check for Hospital Dataset

df_Hospital.isnull().sum()
# 'Unnamed' column is not required for data analysis. Hence we can remove it from dataset

df_Hospital=df_Hospital.drop(['Unnamed: 12','Unnamed: 13' ],axis=1)

round(df_Hospital.isnull().sum()/len(df_Hospital.index)*100,2)
df_Hospital.loc[df_Hospital.isnull().sum(axis=1)>4,:]



df_Hospital=df_Hospital.loc[df_Hospital.isnull().sum(axis=1)<=4,:]

print(df_Hospital.shape)

round(df_Hospital.isnull().sum()/len(df_Hospital.index)*100,2)

df_Hospital.loc[pd.isnull(df_Hospital['NumSubDistrictHospitals_HMIS']),'NumSubDistrictHospitals_HMIS']=0

round(df_Hospital.isnull().sum()/len(df_Hospital.index)*100,2)
df_population.info()
# Population- Null check

round(df_population.isnull().sum()/len(df_Hospital.index)*100,2)
# Merging Population and hospitals





df_population.rename(columns={"State / Union Territory": "State/UT"},inplace=True)

# df_population.head()

df_master = df_population.merge(df_Hospital, on="State/UT" ,how="inner")

df_master.head()



round(df_master.isnull().sum()/len(df_master.index)*100,2)
# Removing string character from Area and Density to convert into float data type

df_master['Area']=df_master['Area'].apply(lambda x : x[:x.index('k')])

df_master['Density']=df_master['Density'].apply(lambda x : x[:x.index('/')])





# conver to int

df_master['Area']=df_master['Area'].str.replace(",", "").apply( lambda x : int(x))

df_master['Density']=df_master['Density'].str.replace(",", "").apply( lambda x : int(x))

df_master.head()
# Deriving Rural and Urban Density for each state



df_master['Rural population_Density'] =   df_master['Rural population'] /df_master['Area']

df_master['Urban population_Density'] =   df_master['Urban population'] /df_master['Area']

df_master.head()
df_m=df_master.drop(['Sno_x','Sno_y'],axis=1)

df_corr=df_m.corr()

fig_dims = (20, 20)

fig, ax = plt.subplots(figsize=fig_dims)

sns.heatmap(df_corr,ax=ax,annot=True)
# Lets get the insights on South India  - Tamil Nadu,Andhra Pradesh, Kerala, Karnataka,Puducherry
a=['Tamil Nadu','Andhra Pradesh', 'Kerala', 'Karnataka','Puducherry']

df_south = df_master.loc[df_master['State/UT'].isin (a)]

df_south
sns.barplot(x="State/UT", y='NumRuralHospitals_NHP18', data=df_south)
sns.barplot(x="State/UT", y='NumUrbanHospitals_NHP18', data=df_south)
sns.barplot(x="State/UT", y='Population', data=df_south)
# calacluate the no of rural and Urban beds per 1000 population

df_south['no_of_rural_beds_per_1000_population'] = df_south['NumRuralBeds_NHP18']/(df_south['Rural population']/1000)

df_south['no_of_urban_beds_per_1000_population'] = df_south['NumUrbanBeds_NHP18']/(df_south['Urban population']/1000)

df_south
sns.barplot(x="State/UT", y='no_of_rural_beds_per_1000_population', data=df_south)



# plt.show()
sns.barplot(x="State/UT", y='no_of_urban_beds_per_1000_population', data=df_south)