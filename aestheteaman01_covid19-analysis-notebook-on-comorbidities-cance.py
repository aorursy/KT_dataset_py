#Data Analyses Libraries

import pandas as pd                

import numpy as np    

from urllib.request import urlopen

import json

import glob

import os



#Importing Data plotting libraries

import matplotlib.pyplot as plt     

import plotly.express as px       

import plotly.offline as py       

import seaborn as sns             

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

import matplotlib.ticker as ticker

import matplotlib.animation as animation



#Other Miscallaneous Libraries

import warnings

warnings.filterwarnings('ignore')

from IPython.display import HTML

import matplotlib.colors as mc

import colorsys

from random import randint

import re
#Importing the clinical spectrum data

clinical_spectrum = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')



#Filtering the data to contain the values only for the confirmed COVID-19 Tests

confirmed = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'

clinical_spectrum = clinical_spectrum[confirmed]



#Viewing the dataset statistics

clinical_spectrum.head()
#Filetering the datasets

positive_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'

positive_condition_spectra = clinical_spectrum[positive_condition]





negative_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'negative'

negative_condition_spectra = clinical_spectrum[negative_condition]



#Taking mean value of the spectra conditions

positive_mean = clinical_spectrum.mean(axis = 0, skipna = True) 

negative_mean = clinical_spectrum.mean(axis = 0, skipna = True) 
#Making columns for the dataset

positive_mean = positive_mean.to_frame()

positive_mean = positive_mean.reset_index()

positive_mean.columns = ['Parameter','Positive_figures']



negative_mean = negative_mean.to_frame()

negative_mean = negative_mean.reset_index()

negative_mean.columns = ['Parameter','Negative_figures']



#Merging both the dataframes together

positive_mean['Negative_figures'] = negative_mean['Negative_figures']



#Viewing the dataset

positive_mean.dropna()

positive_mean.head()
#The most important clinical factors

positive_mean['Change'] =  positive_mean['Positive_figures'] - positive_mean['Negative_figures']

positive_mean.sort_values(['Change'], axis=0, ascending=True, inplace=True) 



#Getting to know the health factors that define HCP Requirement for a patient

lower = positive_mean.head(15)

higher = positive_mean.tail(15)



#Printing the values

for i in lower['Parameter']:

    print('For lower value of {}, the patient is Prone to COVID-19'.format(i))

    

for i in higher['Parameter']:

    print('For higher value of {}, the patient is Prone to COVID-19'.format(i))