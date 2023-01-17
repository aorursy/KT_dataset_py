# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
df_rainfall = pd.read_csv("../input/chennai_reservoir_rainfall.csv")

df_levels = pd.read_csv("../input/chennai_reservoir_levels.csv")
# rainfall dataframe

df_rainfall.head(10)
# Levels dataframe

df_levels.head(10)
# info about our dataframes!

df_levels.info()

df_rainfall.info()

# Check null values for the Data frames!

print(df_rainfall.isnull().sum())

print(df_levels.isnull().sum())
#Let's convert DATE Attribute to Date-Time for better understanding!

df_levels['Date'] = pd.to_datetime(df_levels['Date'],format = '%d-%m-%Y')

df_levels.head()
#Let's convert DATE Attribute to Date-Time for better understanding!

df_rainfall['Date'] = pd.to_datetime(df_rainfall['Date'],format = '%d-%m-%Y')

df_rainfall.tail()
df_levels['Year'] = df_levels['Date'].dt.year

df_levels['Month'] = df_levels['Date'].dt.month

df_levels['Total'] = df_levels['POONDI']+df_levels['REDHILLS']+df_levels['CHEMBARAMBAKKAM']+df_levels['CHOLAVARAM']

grp_by_year = df_levels.groupby('Year')['Total'].sum()

grp_by_year.plot(kind='bar');

plt.ylabel('Level of Water over respective years!');

plt.xticks(rotation=60);
df_rainfall['Year'] = df_rainfall['Date'].dt.year

df_rainfall['Month'] = df_rainfall['Date'].dt.month

df_rainfall['Total'] = df_rainfall['POONDI']+df_rainfall['REDHILLS']+df_rainfall['CHEMBARAMBAKKAM']+df_rainfall['CHOLAVARAM']

grp_by_year = df_rainfall.groupby('Year')['Total'].sum()

grp_by_year.plot(kind='bar');

plt.ylabel('Rainfall of Water over respective years!');

plt.xticks(rotation=60);
df_POONDI = df_levels.loc[:,['POONDI','Year','Total']]

df_REDHILLS = df_levels.loc[:,['REDHILLS','Year','Total']]

df_CHEMBARAMBAKKAM = df_levels.loc[:,['CHEMBARAMBAKKAM','Year','Total']]

df_CHOLAVARAM = df_levels.loc[:,['CHOLAVARAM','Year','Total']]

df_POONDI.head(10)
plt.figure(figsize=(18,12))

plt.subplot(2,2,1)

grp_POONDI = df_POONDI.groupby('Year')['POONDI'].sum()

grp_POONDI.plot(kind='barh');

plt.title("POONDI's Reservoir's performance!");

plt.subplot(2,2,2)

grp_REDHILLS = df_REDHILLS.groupby('Year')['REDHILLS'].sum()

grp_REDHILLS.plot(kind='barh');

plt.title("REDHILLS's Reservoir's performance!");

plt.subplot(2,2,3)

grp_CHEMBARAMBAKKAM = df_CHEMBARAMBAKKAM.groupby('Year')['CHEMBARAMBAKKAM'].sum()

grp_CHEMBARAMBAKKAM.plot(kind='barh');

plt.title("CHEMBARAMBAKKAM's Reservoir's performance!");

plt.subplot(2,2,4)

grp_CHOLAVARAM = df_CHOLAVARAM.groupby('Year')['CHOLAVARAM'].sum()

grp_CHOLAVARAM.plot(kind='barh');

plt.title("CHOLAVARAM's Reservoir's performance!");
# 1st Comes Pondi!

plt.figure(figsize=(15,9))

plt.subplot(1,2,1)

grp_POONDI_comp = df_POONDI.groupby('Year')[['POONDI','Total']].sum()

grp_POONDI_comp.plot(kind='bar',ax=plt.gca());

plt.title("POONDI's Contribution vs Total");

plt.subplot(1,2,2)

grp_POONDI_comp.plot(kind='bar',stacked=True,ax=plt.gca());

plt.title("POONDI's Contribution vs Total in a Stacked Graph!");
# 2nd Comes Redhills!

plt.figure(figsize=(15,9))

plt.subplot(1,2,1)

grp_REDHILLS_comp = df_REDHILLS.groupby('Year')[['REDHILLS','Total']].sum()

grp_REDHILLS_comp.plot(kind='bar',ax=plt.gca());

plt.title("REDHILLS's Contribution vs Total");

plt.subplot(1,2,2)

grp_REDHILLS_comp.plot(kind='bar',stacked=True,ax=plt.gca());

plt.title("REDHILLS's Contribution vs Total in a Stacked Graph!");
# 3rd Comes CHEMBARAMBAKKAM!

plt.figure(figsize=(15,9))

plt.subplot(1,2,1)

grp_CHEMBARAMBAKKAM_comp = df_CHEMBARAMBAKKAM.groupby('Year')[['CHEMBARAMBAKKAM','Total']].sum()

grp_CHEMBARAMBAKKAM_comp.plot(kind='bar',ax=plt.gca());

plt.title("CHEMBARAMBAKKAM's Contribution vs Total");

plt.subplot(1,2,2)

grp_CHEMBARAMBAKKAM_comp.plot(kind='bar',stacked=True,ax=plt.gca());

plt.title("CHEMBARAMBAKKAM's Contribution vs Total in a Stacked Graph!");
# 4th Comes CHOLAVARAM!

plt.figure(figsize=(15,9))

plt.subplot(1,2,1)

grp_CHOLAVARAM_comp = df_CHOLAVARAM.groupby('Year')[['CHOLAVARAM','Total']].sum()

grp_CHOLAVARAM_comp.plot(kind='bar',ax=plt.gca());

plt.title("CHOLAVARAM's Contribution vs Total");

plt.subplot(1,2,2)

grp_CHOLAVARAM_comp.plot(kind='bar',stacked=True,ax=plt.gca());

plt.title("CHOLAVARAM's Contribution vs Total in a Stacked Graph!");
plt.figure(figsize=(15,8))

cor = df_levels.corr()

sns.heatmap(cor,annot=True);

plt.yticks(rotation=30);

plt.xticks(rotation=70);
# rainfall data frame

df_rainfall.head(10)
plt.figure(figsize=(15,7))

rainfall_grp_year = df_rainfall.groupby('Year')['Total'].sum()

rainfall_grp_year.plot(kind='barh');

plt.xlabel('Total Rainfall by combining for all the Reservoirs!');

plt.ylabel('Duration in Years!');

plt.title('Performance/Rainfall level over the Durations on the Reservoirs!');
# We have build a function to perform the similar action on the Rainfall Data frame!

df_POONDI_rainfall = df_rainfall.loc[:,['POONDI','Year','Total']]

df_REDHILLS_rainfall = df_rainfall.loc[:,['REDHILLS','Year','Total']]

df_CHEMBARAMBAKKAM_rainfall = df_rainfall.loc[:,['CHEMBARAMBAKKAM','Year','Total']]

df_CHOLAVARAM_rainfall = df_rainfall.loc[:,['CHOLAVARAM','Year','Total']]

def each_reservoir(data_frame,reservoirs):

    plt.figure(figsize = (15,9))

    plt.subplot(121)

    grp_reservoir_comp = data_frame.groupby('Year')[[reservoirs,'Total']].sum()

    grp_reservoir_comp.plot(kind='bar',ax = plt.gca());

    plt.title(reservoirs);

    plt.xlabel('Rainfall Occured respectively for mentioned Duration!')

    plt.ylabel('Duration in Years');

    plt.subplot(122)

    grp_reservoir_comp.plot(kind = 'bar',stacked=True,ax = plt.gca());    

    plt.title(reservoirs);

    plt.xlabel('Rainfall Occured respectively for mentioned Duration!')

    plt.ylabel('Duration in Years');

    

each_reservoir(df_POONDI_rainfall,'POONDI')

each_reservoir(df_REDHILLS_rainfall,'REDHILLS')

each_reservoir(df_CHEMBARAMBAKKAM_rainfall,'CHEMBARAMBAKKAM')

each_reservoir(df_CHOLAVARAM_rainfall,'CHOLAVARAM')
average_rainfall = df_rainfall['Total'].sum()/len(df_rainfall.index) # this is equivalent to df_rainfall['Total'].mean()

print('Average Rainfall over the Years: {}'.format(average_rainfall))

average_water_levels = df_levels['Total'].sum()/len(df_levels.index) # this is equivalent to df_rainfall['Total'].mean()

print('Average Water Level over the Years: {}'.format(average_water_levels))