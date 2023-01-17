#Importing needed packages

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#Importing data 

dataset = pd.read_csv('../input/insurance/insurance.csv')



#Taking a look at our data

dataset.head()
#Making a distplot of the charges

plt.figure(figsize=(12,4))

sns.distplot(a=dataset['charges'], color='deepskyblue', bins=100)

plt.title('Distribution of the medical charges\n across all the dataset', size='23')

plt.xlabel('Charges',size=18)

plt.show()
#Creating another column containing bins of charges

dataset['charges_bins'] = pd.cut(dataset['charges'], bins=[0, 15000, 30000, 45000, 60000, 75000])



dataset.head()
#Creating a countplot based on the amount of charges

plt.figure(figsize=(12,4))

sns.countplot(x='charges_bins', data=dataset, palette='husl') 

plt.title('Number of people paying x amount\n for each charges category', size='23')

plt.xticks(rotation='25')

plt.ylabel('Count',size=18)

plt.xlabel('Charges',size=18)

plt.show()
#Making a distplot for the age variable

plt.figure(figsize=(12,4))

sns.distplot(a=dataset['age'], color='darkmagenta', bins=100) 

plt.title('Ages distrubution', size='23')

plt.xlabel('Age',size=18)

plt.show()
#Making a lineplot to check if there is any correlation between age and charges

plt.figure(figsize=(12,4))

sns.lineplot(x='age', y='charges', data=dataset, color='mediumvioletred')

plt.title('Charges according to age', size='23')

plt.ylabel('Charges',size=18)

plt.xlabel('Ages',size=18)

plt.show()
#Making bins for the ages

dataset['age_bins'] = pd.cut(dataset['age'], bins = [0, 20, 35, 50, 70])



#Creating boxplots based on the amount of diffrent age categories

plt.figure(figsize=(12,4))

sns.boxplot(x='age_bins', y='charges', data=dataset, palette='RdPu') 

plt.title('Charges according to age categories', size='23')

plt.xticks(rotation='25')

plt.grid(True)

plt.ylabel('Charges',size=18)

plt.xlabel('Age',size=18)

plt.show()
#Countplot of males/females

plt.figure(figsize=(12,4))

sns.countplot(x='sex', data=dataset, palette='PuBu') 

plt.title('Number of males/females', size='23')

plt.ylabel('Count',size=18)

plt.xlabel('Sex',size=18)

plt.show()
#Cheking the charges distributions for males and females

x1 = sns.FacetGrid(dataset, row='sex', height=4, aspect=3.5)

x1 = x1.map(sns.distplot, 'charges', color='cornflowerblue')

plt.show()
#Making a distplot for our BMI variable 

plt.figure(figsize=(12,4))

sns.distplot(a=dataset['bmi'], color='mediumseagreen', bins=100)

plt.title('Distribution of the BMI variable\n across all the dataset', size='23')

plt.xlabel('BMI',size=18)

plt.show()
#Scatterplot to check for correlation 

plt.figure(figsize=(12,4))

sns.scatterplot(x='bmi', y='charges', data=dataset, color='seagreen')

plt.title('Charges according to BMI', size='23')

plt.ylabel('Charges',size=18)

plt.xlabel('BMI',size=18)

plt.show()
#Making bins and labels for the BMI

bins = [0, 18.5, 25, 30, 35, 40, 60]

labels = ['Underweight', 'Average', 'Overweight', 'Obese 1', 'Obese 2', 'Obese 3']

dataset['bmi_bins'] = pd.cut(dataset['bmi'], bins=bins, labels=labels)



#Checking the charges according to BMI 

plt.figure(figsize=(12,4))

sns.barplot(x='bmi_bins', y='charges', data=dataset, palette='Greens')

plt.title('Charges according to BMI categories', size='23')

plt.ylabel('Charegs',size=18)

plt.xlabel('BMI categories',size=18)

plt.show()
#Countplot for diffrent 'number of children' categories

plt.figure(figsize=(12,4))

sns.countplot(x='children', data=dataset, palette='YlGnBu') 

plt.title('Number of pepople having x children', size='23')

plt.ylabel('Count',size=18)

plt.xlabel('Number of children',size=18)

plt.show()
#Creating a violinplot for each category

plt.figure(figsize=(12,4))

sns.violinplot(x='children', y='charges', data=dataset, hue='sex', palette='YlGnBu')

plt.title('Charges according to number of children', size='23')

plt.ylabel('Charges',size=18)

plt.xlabel('Number of children',size=18)

plt.show()
#Countplot to compare the number of smokers and non-smokers

plt.figure(figsize=(12,4))

sns.countplot(x='smoker', data=dataset, hue='sex', palette='YlOrBr') 

plt.title('Number of smokers and non-smokers', size='23')

plt.ylabel('Count',size=18)

plt.xlabel('Smoker',size=18)

plt.show()
#Creating boxplots to compare charges distributions for smokers and non-smokers

plt.figure(figsize=(12,12))

ax1 = plt.subplot2grid((2,1),(0,0))

ax2 = plt.subplot2grid((2,1),(1,0))

sns.boxplot(x='charges', y='sex' ,data=dataset[dataset['smoker']=='yes'], palette='YlOrBr', ax=ax1)

ax1.set_title('Somker',size='23')

ax1.set_ylabel('Sex',size=18)

ax1.set_xlabel('Charges',size=18)

sns.boxplot(x='charges', y='sex' ,data=dataset[dataset['smoker']=='no'], palette='YlOrBr', ax=ax2)

ax2.set_title('Non-somker', size='23')

ax2.set_ylabel('Sex',size=18)

ax2.set_xlabel('Charges',size=18)

plt.tight_layout()

plt.show()
#Creating a FacetGrid to compare charges of smokers to non-smoker charges with diffrent BMI categories

x2 = sns.FacetGrid(dataset, row='smoker', height=4, aspect=3.5)

x2 = x2.map(sns.barplot, 'bmi_bins', 'charges', palette='YlOrBr', order=labels)

plt.show()
#Countplot to compare the number of individuals from diffrent regions

plt.figure(figsize=(12,4))

sns.countplot(x='region', data=dataset, palette='husl') 

plt.title('Number of individuals from diffrent regions', size='23')

plt.ylabel('Count',size=18)

plt.xlabel('Region',size=18)

plt.show()
#Creating distplots to compare charges distributions for diffrent regions and the overall dsitribution of charges

plt.figure(figsize=(12,12))

ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)

ax2 = plt.subplot2grid((3,2),(1,0))

ax3 = plt.subplot2grid((3,2),(1,1))

ax4 = plt.subplot2grid((3,2),(2,0))

ax5 = plt.subplot2grid((3,2),(2,1))

sns.distplot(a=dataset['charges'], ax=ax1,  color='lime')

ax1.set_title('Overall distribution',size='23')

ax1.set_xlabel('Charges',size=18)



axis = [ax2, ax3, ax4, ax5]

regions = ['southwest', 'southeast', 'northwest', 'northeast']

for axe, region in zip(axis, regions):

    data = dataset[dataset['region']==region]

    sns.distplot(a=data['charges'], ax=axe,  color='darkorchid')

ax2.set_title('Southwest', size='23')

ax2.set_xlabel('Charges',size=18)

ax3.set_title('Southeast', size='23')

ax3.set_xlabel('Charges',size=18)

ax4.set_title('Northwest', size='23')

ax4.set_xlabel('Charges',size=18)

ax5.set_title('Northeast', size='23')

ax5.set_xlabel('Charges',size=18)

plt.tight_layout()

plt.show()