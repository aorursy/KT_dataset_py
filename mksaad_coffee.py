# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd



cofee_data=pd.read_csv("../input/coffee-and-code/CoffeeAndCodeLT2018.csv")
cofee_data.head()
cofee_data
cofee_data.describe()
cofee_data.columns
#to find the maximum value in CoffeeCupsPerDay (maximum number of cups of cofee per day)

cofee_data.describe().CoffeeCupsPerDay.loc['max']
cofee_data.Gender.value_counts()
cofee_data.groupby('Gender').CoffeeCupsPerDay.sum()
cofee_data.groupby('Gender').Gender.count()
cofee_data.groupby('CoffeeType').CoffeeType.count().sort_values(ascending=False)
cofee_data.groupby('AgeRange').AgeRange.count().sort_values(ascending=False)
cofee_data.groupby('CodingWithoutCoffee').CodingWithoutCoffee.count()
cofee_data.groupby('CoffeeSolveBugs').CoffeeSolveBugs.count().sort_values(ascending=False)
#to count the number of null/missing value

cofee_data.isnull().sum()
cofee_data.CoffeeType .fillna("Nescafe",inplace=True)
cofee_data.AgeRange.fillna("18 to 29",inplace=True)
cofee_data.isnull().sum()
cofee_data.shape
cofee_data_mean=cofee_data.mean()

print(cofee_data_mean)
cofee_data.CoffeeCupsPerDay.dtype
sns.pairplot(cofee_data)
#scatter plot

sns.scatterplot(x=cofee_data['CodingHours'], y=cofee_data['CoffeeCupsPerDay'])
#The scatterplot above suggests that CofeeCupsPerDay and CodingHours are positively correlated,

#where programmers writes more coding hours also drink more cups of cofee.
#To double-check the strength of this relationship,add a regression line.
#Scatter plot regression line showing the relationship between 'CodingHours' and 'CoffeeCupsPerDay'

sns.regplot(x=cofee_data['CodingHours'], y=cofee_data['CoffeeCupsPerDay'])
sns.scatterplot(x=cofee_data['CodingHours'], y=cofee_data['CoffeeCupsPerDay'],hue=cofee_data['CoffeeSolveBugs'])
#This scatter plot shows This scatter plot shows how cups of cofee increase/help solve bugs
#we can use the sns.lmplot command to add three regression lines, corresponding to CofeeSolveBugs, sometimes and CofeeNOTSolveBugs.

sns.lmplot(x="CodingHours",y="CoffeeCupsPerDay",hue="CoffeeSolveBugs",data=cofee_data)
sns.scatterplot(x=cofee_data['CodingHours'], y=cofee_data['CoffeeCupsPerDay'],hue=cofee_data['Gender'])
sns.lmplot(x="CodingHours",y="CoffeeCupsPerDay",hue="Gender",data=cofee_data)
#According to the plot above, the regression line has a slightly positive slope, 

#this tells us that there is a slightly positive correlation between 'CodingHours' and 'CofeeCupsPerDay'.

#Thus, male have a slight preference to drink cofee during coding hours.
sns.pairplot(cofee_data, hue='CoffeeTime')
sns.lmplot(x="CodingHours",y="CoffeeCupsPerDay",hue="CodingWithoutCoffee",data=cofee_data)
plt.figure(figsize=(18,8))

plt.title("Cofee Type")

sns.countplot(x="CoffeeType",data=cofee_data)

plt.figure(figsize=(16,7))

plt.title("Cofee Time")

sns.countplot(x="CoffeeTime",data=cofee_data)
plt.figure(figsize=(16,8))

plt.title("Age Range")

sns.countplot(x="AgeRange",data=cofee_data,hue="Gender")

plt.figure(figsize=(16,6))

plt.title("Coding Without Coffee")

sns.countplot(x="CodingWithoutCoffee",data=cofee_data,hue="Gender")
