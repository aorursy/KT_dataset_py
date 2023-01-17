# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt   

import seaborn as sns 

path ="../input"

os.chdir(path)

ds = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1")





# Data shape

ds.shape  



# Read top rows

ds.head()

# Read bottom rows



# overview of data

ds.info()



# overview of district and crimes observed



desc_order = ds['DISTRICT'].value_counts().index

sns.countplot("DISTRICT", data = ds, order = desc_order)







# Another way to display the crimes per district





plt.figure(figsize=(20,8))

ds['DISTRICT'].value_counts().plot.bar()

plt.title('District wise Crimes')

plt.ylabel('No of Crimes')

plt.xlabel('Police District')



plt.show()

# Maxium number of Crimes observed in Police District B2 in Boston.







# Year wise Crimes in Boston



plt.figure(figsize=(20,8))

sns.countplot(x='YEAR', data = ds)

plt.ylabel('No of Crimes')

plt.title('Year wise Crimes Reported')

plt.show()

# Max number of Crimes happened in 2017











# Top 10 different types of Crime

plt.figure(figsize=(16,8))

top10ctype = ds.groupby('OFFENSE_CODE_GROUP')['INCIDENT_NUMBER'].count().sort_values(ascending=False)

top10ctype = top10ctype [:10]

top10ctype.plot(kind='bar', color='red')

plt.ylabel('Number of Crimes')

plt.title('Top 10 Types of Crime')



plt.show()





#year wise breakup of Crimes by District

sns.catplot(x="DISTRICT",       

            hue="MONTH",      

            col="YEAR",       

            data=ds,

            kind="count")