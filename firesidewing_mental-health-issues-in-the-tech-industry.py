import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import rcParams

import math

import seaborn as sb
%matplotlib inline

rcParams['figure.figsize'] = 13, 6

sb.set()
data = pd.read_csv('../input/survey.csv')



#Converting all strings in 'gender' column to uppercase to makes things easier to deal with

data['Gender'] = data['Gender'].str.upper()



#can = data[data['Country'].str.contains('Can')].reset_index()



#female = data[data['Gender'].str.contains('FE')].reset_index()



print('A total of {} people surveyed'.format(data.shape[0]))
#plt.margins(0.15)

plt.xticks(rotation='vertical');

sb.countplot(data['Gender']);
# Here I clean up the gender column a bit by replacing spelling mistakes with what i imagine their inent is

# Also i put the genders into 3 neat catagories for easier analysis



data['Gender'].loc[data.Gender.isin(['ENBY','SOMETHING KINDA MALE?','QUEER/SHE/THEY','NON-BINARY','NAH','ALL','FLUID',

                                     'GENDERQUEER','ANDROGYNE','AGENDER','NEUTER','QUEER','A LITTLE ABOUT YOU','P',

                                     'OSTENSIBLY MALE, UNSURE WHAT THAT REALLY MEANS','TRANS-FEMALE','TRANS WOMAN','FEMALE (TRANS)'])] = 'OTHER'

data['Gender'].loc[data.Gender.isin(['M','MALE-ISH','MAILE','CIS MALE','MAL','MALE (CIS)','MAKE','GUY (-ISH) ^_^','MALE LEANING ANDROGYNOUS',

                                     'MALE ','MAN','MSLE','MAIL','MALR','CIS MAN'])] = 'MALE'

data['Gender'].loc[data.Gender.isin(['CIS FEMALE','F','WOMAN','FEMAKE','FEMALE ','CIS-FEMALE/FEMME',

                                     'FEMALE (CIS)','FEMAIL'])] = 'FEMALE'
rcParams['figure.figsize'] = 8, 6

sb.countplot(data['Gender']);
rcParams['figure.figsize'] = 13, 6

sb.countplot(data['Country']);

plt.xticks(rotation='vertical');
#can = data.loc[data['Country']=='Canada']

#us = data.loc[data['Country']=='United States']

#uk = data.loc[data['Country']=='United Kingdom']

countries = pd.concat([data.loc[data['Country']=='Canada'], data.loc[data['Country']=='United States'], data.loc[data['Country']=='United Kingdom']]).reset_index(drop=True)

print('There consists {} people from the top 3 countries out of the {} people surveyed'.format(countries.shape[0], data.shape[0]))
rcParams['figure.figsize'] = 13, 6

sb.countplot(x=countries['work_interfere'], order=['Never', 'Rarely', 'Sometimes', 'Often']);

plt.xlabel('If you have a mental health condition, do you feel that it interferes with your work?');
work_sum = countries['work_interfere'].value_counts().reset_index()

more_than_never = work_sum['work_interfere'][0] + work_sum['work_interfere'][2] + work_sum['work_interfere'][3]

print('{} people, or {:.1%}, believe that their mental health condition interferes with their work either sometimes or more'.format(more_than_never, more_than_never/countries.shape[0]))

print('With {} ({:.1%}) people saying it intereferes often'.format(work_sum['work_interfere'][3], work_sum['work_interfere'][3]/countries.shape[0]))
rcParams['figure.figsize'] = 8, 6

sb.countplot(countries['treatment']);

plt.xlabel('Have you sought treatment for a mental health condition?');
treatment_count = countries['treatment'].value_counts().reset_index()

print('Luckily {} ({:.1%}) have sought treatment for their mental health issues'.format(treatment_count['treatment'][0], treatment_count['treatment'][0]/countries.shape[0]))
male = countries.loc[countries['Gender']=='MALE']

male_treatment = male.loc[male['treatment']=='Yes'].reset_index(drop=True)



female = countries.loc[countries['Gender']=='FEMALE']

female_treatment = female.loc[female['treatment']=='Yes'].reset_index(drop=True)



other = countries.loc[countries['Gender']=='OTHER']

other_treatment = other.loc[other['treatment']=='Yes'].reset_index(drop=True)



print('Out of {} males surveyed, {} ({:.1%}) sought treatment'.format(male.shape[0], male_treatment.shape[0], male_treatment.shape[0]/male.shape[0]))

print('Out of {} females surveyed, {} ({:.1%}) sought treatment'.format(female.shape[0], female_treatment.shape[0], female_treatment.shape[0]/female.shape[0]))

print('Out of {} people who identify as anything other than male or female surveyed, {} ({:.1%}) sought treatment'.format(other.shape[0], other_treatment.shape[0], other_treatment.shape[0]/other.shape[0]))