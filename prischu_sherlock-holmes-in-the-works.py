#Libraries required for Data Analysis

import pandas as pd

import numpy as np

import random as rnd



#Libraries required for Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Libraries required for Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
#Load in the Dataset

dataset = pd.read_csv("../input/database.csv")
#Display features available in the Dataset

print(dataset.columns.values)
#Preview the Data (Beginning)

pd.set_option('display.max_columns', None)

dataset.head()
#Preview the Data (Ending)

dataset.tail()
#Display Feature Types

dataset.info()
#Convert Categorical Features that should be numerics for analysis

dataset['Perpetrator Age'] = pd.to_numeric(dataset['Perpetrator Age'], errors='coerce')
#Review the distribution of the numerical features

dataset.describe()
#Review the distribution of the categorial features

dataset.describe(include=['O'])
#Analyzing Unique Values in the Dataset to make assumptions

print(dataset.State.unique())

print('-' * 60)

print(dataset['Crime Type'].unique())

print('-' * 60)

print(dataset['Weapon'].unique())

print('-' * 60)

print(dataset['Victim Ethnicity'].unique())

print('-' * 60)

print(dataset['Victim Race'].unique())

print('-' * 60)

print(dataset['Victim Age'].unique())
#Dataset Filtered to exclude the Perpetrator Sex 'Unknown'

bins = np.arange(0, 100, 5)

datasetNoUnknown = dataset.loc[(dataset['Perpetrator Sex'] != 'Unknown')]

grid = sns.FacetGrid(datasetNoUnknown, row='Crime Type', col='Perpetrator Sex', size=3.3, aspect=2)

grid.map(plt.hist,'Perpetrator Age', bins = bins)

grid.add_legend()
#Dataset Filtered to exclude Perpetrator Sex 'Unknown' and Crime Type 'Manslaughter by Negligence'

bins = np.arange(0, 100, 5)

datasetNoUnknown = dataset.loc[(dataset['Perpetrator Sex'] != 'Unknown') & (dataset['Crime Type'] != 'Manslaughter by Negligence')]

grid = sns.FacetGrid(datasetNoUnknown, row='Perpetrator Race', col='Perpetrator Sex', size=3.3, aspect=2)

grid.map(plt.hist,'Perpetrator Age', bins = bins)

grid.add_legend()
#Dataset Filtered to exclude Perpetrator Sex 'Unknown' and Crime Type 'Murder or Manslaughter'

bins = np.arange(0, 100, 5)

datasetNoUnknown = dataset.loc[(dataset['Perpetrator Sex'] != 'Unknown') & (dataset['Crime Type'] != 'Murder or Manslaughter')]

grid = sns.FacetGrid(datasetNoUnknown, row='Perpetrator Race', col='Perpetrator Sex', size=3.3, aspect=2)

grid.map(plt.hist,'Perpetrator Age', bins = bins)

grid.add_legend()
#State with the highest amount of historical homicide crimes

#Full Dataset

dataset.groupby(["State"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#State with the highest Crime Type 'Murder or Manslaughter'

temp1 = dataset.loc[(dataset['Crime Type'] != 'Manslaughter by Negligence')]

temp1.groupby(["State"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#State with the highest Crime Type 'Manslaughter by Negligence'

temp2 = dataset.loc[(dataset['Crime Type'] != 'Murder or Manslaughter')]

temp2.groupby(["State"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#Weapon most likely used for homicides

dataset.groupby(["Perpetrator Sex", "Weapon"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#Years with the highest amount of Homicide Crimes reported

g = sns.factorplot("Year", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Age different Sexes are most likely victims at

bins = np.arange(0, 100, 5)

grid = sns.FacetGrid(datasetNoUnknown, col='Victim Sex', size=3.3, aspect=2)

grid.map(plt.hist,'Victim Age', bins = bins)

grid.add_legend()
#Highest Homicide Rate between which two sexes

# Set a default value

dataset['PVSex'] = 'Unknown kills ?'



#Value for Female x Female

dataset['PVSex'][(dataset['Perpetrator Sex'] == 'Female') & (dataset['Victim Sex'] == 'Female')] = 'Female kills Female'

 

#Value for Female x Male

dataset['PVSex'][(dataset['Perpetrator Sex'] == 'Female') & (dataset['Victim Sex'] == 'Male')] = 'Female kills Male'



#Value for Male x Male

dataset['PVSex'][(dataset['Perpetrator Sex'] == 'Male') & (dataset['Victim Sex'] == 'Male')] = 'Male kills Male'



#Value for Male x Female

dataset['PVSex'][(dataset['Perpetrator Sex'] == 'Male') & (dataset['Victim Sex'] == 'Female')] = 'Male kills Female'
g = sns.factorplot("PVSex", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Highest Homicide Ethnicity Victim

g = sns.factorplot("Victim Ethnicity", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Highest Homicide Race Victim

g = sns.factorplot("Victim Race", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Homicide more likely between same or different ethnicities

# Set a default value

dataset['EthK'] = 'Unknown kills ?'



#Value for Not Hispanic x Not Hispanic

dataset['EthK'][(dataset['Perpetrator Ethnicity'] == 'Not Hispanic') & (dataset['Victim Ethnicity'] == 'Not Hispanic')] = 'Not Hispanic kills Not Hispanic'

 

#Value for Not Hispanic x Hispanic

dataset['EthK'][(dataset['Perpetrator Ethnicity'] == 'Not Hispanic') & (dataset['Victim Ethnicity'] == 'Hispanic')] = 'Not Hispanic kills Hispanic'



#Value for Hispanic x Not Hispanic

dataset['EthK'][(dataset['Perpetrator Ethnicity'] == 'Hispanic') & (dataset['Victim Ethnicity'] == 'Not Hispanic')] = 'Hispanic kills Not Hispanic'



#Value for Hispanic x Hispanic

dataset['EthK'][(dataset['Perpetrator Ethnicity'] == 'Hispanic') & (dataset['Victim Ethnicity'] == 'Hispanic')] = 'Hispanic kills Hispanic'
g = sns.factorplot("EthK", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Most Homicides occur during which months

g = sns.factorplot("Month", data=dataset, aspect=1.5, kind="count")

g.set_xticklabels(rotation=90)
#Most Used Weapon per State

dataset.groupby(["State", "Weapon"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#Homicide more likely between same or different races

# Set a default value

dataset['RaceK'] = 'Unknown kills ?'



#Value for Native American/Alaska Native x Native American/Alaska Native

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Native American/Alaska Native') & (dataset['Victim Race'] == 'Native American/Alaska Native')] = 'Native American/Alaska Native kills Native American/Alaska Native'

 

#Value for Native American/Alaska Native x White

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Native American/Alaska Native') & (dataset['Victim Race'] == 'White')] = 'Native American/Alaska Native kills White'



#Value for Native American/Alaska Native x Black

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Native American/Alaska Native') & (dataset['Victim Race'] == 'Black')] = 'Native American/Alaska Native kills Black'



#Value for Native American/Alaska Native x Asian/Pacific Islander

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Native American/Alaska Native') & (dataset['Victim Race'] == 'Asian/Pacific Islander')] = 'Native American/Alaska Native kills Asian/Pacific Islander'



#Value for White x Native American/Alaska Native

dataset['RaceK'][(dataset['Perpetrator Race'] == 'White') & (dataset['Victim Race'] == 'Native American/Alaska Native')] = 'White kills Native American/Alaska Native'

 

#Value for White x White

dataset['RaceK'][(dataset['Perpetrator Race'] == 'White') & (dataset['Victim Race'] == 'White')] = 'White kills White'



#Value for White x Black

dataset['RaceK'][(dataset['Perpetrator Race'] == 'White') & (dataset['Victim Race'] == 'Black')] = 'White kills Black'



#Value for White x Asian/Pacific Islander

dataset['RaceK'][(dataset['Perpetrator Race'] == 'White') & (dataset['Victim Race'] == 'Asian/Pacific Islander')] = 'White kills Asian/Pacific Islander'



#Value for Black x Native American/Alaska Native

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Black') & (dataset['Victim Race'] == 'Native American/Alaska Native')] = 'Black kills Native American/Alaska Native'

 

#Value for Black x White

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Black') & (dataset['Victim Race'] == 'White')] = 'Black kills White'



#Value for Black x Black

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Black') & (dataset['Victim Race'] == 'Black')] = 'Black kills Black'



#Value for Black x Asian/Pacific Islander

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Black') & (dataset['Victim Race'] == 'Asian/Pacific Islander')] = 'Black kills Asian/Pacific Islander'



#Value for Asian/Pacific Islander x Native American/Alaska Native

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Asian/Pacific Islander') & (dataset['Victim Race'] == 'Native American/Alaska Native')] = 'Asian/Pacific Islander kills Native American/Alaska Native'

 

#Value for Asian/Pacific Islander x White

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Asian/Pacific Islander') & (dataset['Victim Race'] == 'White')] = 'Asian/Pacific Islander kills White'



#Value for Asian/Pacific Islander x Black

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Asian/Pacific Islander') & (dataset['Victim Race'] == 'Black')] = 'Asian/Pacific Islander kills Black'



#Value for Asian/Pacific Islander x Asian/Pacific Islander

dataset['RaceK'][(dataset['Perpetrator Race'] == 'Asian/Pacific Islander') & (dataset['Victim Race'] == 'Asian/Pacific Islander')] = 'Asian/Pacific Islander kills Asian/Pacific Islander'
g = sns.factorplot("RaceK", data=dataset, aspect=2, kind="count")

g.set_xticklabels(rotation=90)