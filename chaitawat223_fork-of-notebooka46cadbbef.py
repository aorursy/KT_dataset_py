import pandas as pd

import numpy as np

import seaborn as sns

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

%matplotlib inline
outbreak = pd.read_csv('outbreaks_kaggle.csv')
outbreak.info()
#sns.factorplot('Month',data=outbreak,kind='count')
outbreak.head()
#sns.factorplot('Year',data=outbreak,kind='count')
#sns.factorplot(x='Year',y='Fatalities',data=outbreak,kind='point')
out_year = outbreak.groupby(outbreak['Year'])



out_year.describe().T
ill_year = outbreak['Fatalities'].groupby(outbreak['Year'])



ill_year.sum()
#number of patients each year



illness_by_year = outbreak['Illnesses'].groupby(outbreak['Year'])



illness_by_year.sum().plot(kind='bar')

#count of death each year



death_by_year = outbreak['Fatalities'].groupby(outbreak['Year'])



death_by_year.sum().plot(kind='bar')
#count of patients adminning to hospital



hospitalize_by_year = outbreak['Hospitalizations'].groupby(outbreak['Year'])



hospitalize_by_year.sum().plot(kind='bar')
rep_1998 = outbreak[outbreak['Year']==1998]
rep_1998.head()
#number of species causing morbidity



plt.figure(figsize=(10,6))

rep_1998['Illnesses'].groupby(rep_1998['Species']).sum().plot(kind='bar')
#Number of patients in each state



plt.figure(figsize=(10,6))

rep_1998['Illnesses'].groupby(rep_1998['State']).sum().plot(kind='bar')
#Number of patients by month



plt.figure(figsize=(10,6))

rep_1998['Illnesses'].groupby(rep_1998['Month']).sum().plot(kind='bar')
#Number of death case by species



plt.figure(figsize=(10,6))

rep_1998['Fatalities'].groupby(rep_1998['Species']).sum().plot(kind='bar')
#rep_1998_cm[rep_1998_cm['Species']=='Amnesic shellfish poison']
rep_2004 = outbreak[outbreak['Year']==2004]
rep_2004.head()
#number of species causing morbidity



plt.figure(figsize=(10,6))

rep_2004['Illnesses'].groupby(rep_2004['Species']).sum().plot(kind='bar')
#Number of patients in each state



plt.figure(figsize=(10,6))

rep_2004['Illnesses'].groupby(rep_2004['State']).sum().plot(kind='bar')
#Number of patients by month



plt.figure(figsize=(10,6))

rep_2004['Illnesses'].groupby(rep_2004['Month']).sum().plot(kind='bar')
#Number of death case by species



plt.figure(figsize=(10,6))

rep_2004['Fatalities'].groupby(rep_2004['Species']).sum().plot(kind='bar')
rep_2011 = outbreak[outbreak['Year']==2011]



rep_2011.head()
#number of species causing morbidity



plt.figure(figsize=(10,6))

rep_2011['Illnesses'].groupby(rep_2011['Species']).sum().plot(kind='bar')
#Number of patients in each state



plt.figure(figsize=(10,6))

rep_2011['Illnesses'].groupby(rep_2011['State']).sum().plot(kind='bar')
#Number of patients by month



plt.figure(figsize=(10,6))

rep_2011['Illnesses'].groupby(rep_2011['Month']).sum().plot(kind='bar')
#Number of death case by species



plt.figure(figsize=(10,6))

rep_2011['Fatalities'].groupby(rep_2011['Species']).sum().plot(kind='bar')
rep_2015 = outbreak[outbreak['Year']==2015]



rep_2015.head()
#number of species causing morbidity



plt.figure(figsize=(10,6))

rep_2015['Illnesses'].groupby(rep_2015['Species']).sum().plot(kind='bar')
#Number of patients in each state



plt.figure(figsize=(10,6))

rep_2015['Illnesses'].groupby(rep_2015['State']).sum().plot(kind='bar')
#Number of patients by month



plt.figure(figsize=(10,6))

rep_2011['Illnesses'].groupby(rep_2011['Month']).sum().plot(kind='bar')
#Number of death case by species



plt.figure(figsize=(10,6))

rep_2011['Fatalities'].groupby(rep_2011['Species']).sum().plot(kind='bar')