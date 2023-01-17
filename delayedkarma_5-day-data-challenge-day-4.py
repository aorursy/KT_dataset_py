# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# We look at the Crime Data for Los Angeles 2010-2017

df = pd.read_csv('../input/Crime_Data_2010_2017.csv')

df.head()
df.info()
# First let's visualize the number of crimes according to area name

fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(y="Area Name",data=df,palette='bright',alpha=0.75)

ax.set_title('Crime in LA (2010-17) by Area');
df['Victim Sex'] = df["Victim Sex"].replace(['H','-'],'X')

df['Victim Sex'].value_counts()

# We had some weird entries for Victim Sex, like 'H' or '-', replacing those with'X'(Unknown)
# Remve null values for victim sex

df=df[df['Victim Sex'].notnull()]

df.shape
# We only look at crime data from the 5 highest crime areas

areas = [area for area in df['Area Name'].value_counts().head().reset_index()['index']]

df2 = pd.DataFrame()

for area in areas:

    df2 = pd.concat([df2,df[df['Area Name']==area]],axis=0)

df2.head()    
df2['Area Name'].value_counts() # Checks out
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(x='Area Name',hue='Victim Sex',data=df2,palette='bright',alpha=0.75)

ax.set_title('Victims, by Sex, in the highest crime neighbourhoods of Los Angeles (2010-17)')

ax.set_xlabel('Area Name', fontsize=15);

# What inferences can we draw from this? Why does violence against women spike in certain 

# neighbourhoods? Looking at the nature of the crime (provided in the dataset) or the

# neighbourhood incomes and demographics (which would have to be sourced from outside), might

# provide some clues. 