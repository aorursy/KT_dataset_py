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
from matplotlib.ticker import MaxNLocator

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
hospitalized_condtion = clinical_spectrum['patient_addmited_to_regular_ward_1_yes_0_no'] == 't'
us_hospitalized_spectra = clinical_spectrum[hospitalized_condtion]


unhospitalized_condtion = clinical_spectrum['patient_addmited_to_regular_ward_1_yes_0_no'] == 'f'
us_unhospitalized_spectra = clinical_spectrum[unhospitalized_condtion]

#Taking mean value of the spectra conditions
hospitalized_mean = us_hospitalized_spectra.mean(axis = 0, skipna = True) 
unhospitalized_mean = us_unhospitalized_spectra.mean(axis = 0, skipna = True) 

#Making columns for the dataset
hospitalized_mean = hospitalized_mean.to_frame()
hospitalized_mean = hospitalized_mean.reset_index()
hospitalized_mean.columns = ['Parameter','Hospitalized_figures']

unhospitalized_mean = unhospitalized_mean.to_frame()
unhospitalized_mean = unhospitalized_mean.reset_index()
unhospitalized_mean.columns = ['Parameter','Unhospitalized_figures']

#Merging both the dataframes together
hospitalized_mean['Unhospitalized_figures'] = unhospitalized_mean['Unhospitalized_figures']

#Viewing the dataset
hospitalized_mean.dropna()

#The most important clinical factors
hospitalized_mean['Change'] =  hospitalized_mean['Hospitalized_figures'] - hospitalized_mean['Unhospitalized_figures']
hospitalized_mean.sort_values(['Change'], axis=0, ascending=True, inplace=True) 

#Getting to know the health factors that define HCP Requirement for a patient
lower = hospitalized_mean.head(10)
higher = hospitalized_mean.tail(10)
#Printing the values
for i in lower['Parameter']:
    print('For lower value of {}, the patient may require HCP'.format(i))
    
for i in higher['Parameter']:
    print('For higher value of {}, the patient may require HCP'.format(i))