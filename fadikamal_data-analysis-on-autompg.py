import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 
data=pd.read_csv("../input/auto-mpg.csv")
data
data.columns ## Get Columns names
# Get The shape of the data 

data.shape
# Get the top Five rows 

data.head()
# Get The Final rows 

data.tail()
# Get description of the data using info method 

data.info() #there's no missing data
# Get List of numerical attributes

data.select_dtypes(exclude='object').columns
# Get the summary with upto 2 decimals and call transpose for a better view

data.select_dtypes(exclude='object').describe().round(decimals=2).transpose()
# Get the categorical attributes

data.select_dtypes(include='object').columns
data.select_dtypes(include=['object']).describe().transpose()
data['mpg'].describe()
#Get the distribution plot

sns.distplot(data['mpg'])
print(data['mpg'].mean(),data['mpg'].mode(),data['mpg'].median())
# what about the numerical attributes

num_attributes = data.select_dtypes(exclude='object').drop(['mpg'], axis=1).copy()

len(num_attributes.columns)
fig=plt.figure(figsize=(12,18))

for i in range(len(num_attributes.columns)):

    fig.add_subplot(2,3,i+1)

    sns.distplot(num_attributes.iloc[:,i].dropna(),hist=False,rug=True)
## Get The Boxplot to know about the Outliers

for i in range(6):

    fig.add_subplot(2,3,i+1)

    sns.boxplot(y=num_attributes.iloc[:,i],data=data)

    
# The correlation of numerical attributes

corr=data.corr()

corr #The top influencers are cylinders,displacements and weight
# Get scatter plot of the influencers with Target

col=['mpg','cylinders','displacement','weight']

sns.pairplot(data[col])
# Lets see the correlation between the influencers 

friends=['displacement','weight']

influencers=data[friends]

influencers.corr() # we sea that the correlation with the influencers  is very high so we need to drop one on them 
import pandas as pd

auto_mpg = pd.read_csv("../input/auto-mpg.csv")