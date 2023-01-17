import numpy as np 

import pandas as pd 

import os

import math

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.core import display as ICD

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 5000)

base_dir = '../input/'

fileName = 'multipleChoiceResponses.csv'

filePath = os.path.join(base_dir,fileName)

survey2_expanded = pd.read_csv(filePath) 

responsesOnly_expanded = survey2_expanded[1:]

surveySchemaPath = os.path.join(base_dir,'SurveySchema.csv')

surveySchema = pd.read_csv(surveySchemaPath)

multipleChoicePath = os.path.join(base_dir,'multipleChoiceResponses.csv')

multipleChoice = pd.read_csv(multipleChoicePath)

responsesOnly = multipleChoice[1:]

columns_to_keep2 = ['Time from Start to Finish (seconds)']

responsesOnlyDuration = responsesOnly[columns_to_keep2]

responsesOnlyDuration = pd.to_numeric(responsesOnlyDuration['Time from Start to Finish (seconds)'], errors='coerce')

responsesOnlyDuration = responsesOnlyDuration/60

responsesOnlyDuration = pd.DataFrame(responsesOnlyDuration)

responsesOnlyDuration.columns = ['Time from Start to Finish (minutes)']

responsesOnlyDuration = responsesOnlyDuration['Time from Start to Finish (minutes)']

sns.distplot(responsesOnlyDuration,bins=5000).set(xlim=(0, 60))

print('Average Time Spent Taken for Survey: 15-20min')
print('Total Number of Responses: ',responsesOnly_expanded.shape[0])

responsesOnly_expanded2 = responsesOnly_expanded

responsesOnly_expanded2 = pd.to_numeric(responsesOnly_expanded2['Time from Start to Finish (seconds)'], errors='coerce')

responsesOnly_expanded2 = pd.DataFrame(responsesOnly_expanded2)

responsesOnly_expanded2 = responsesOnly_expanded2[responsesOnly_expanded2['Time from Start to Finish (seconds)'] > 600]  

print('Total Number of Respondents That Took More Than 10 Minutes: ',responsesOnly_expanded2.shape[0])
Dir = '../input/'

freeform_df = pd.read_csv(Dir + 'freeFormResponses.csv', low_memory=False, header=[0,1])

multi_df = pd.read_csv(Dir + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])

SurveySchema = pd.read_csv(Dir + 'SurveySchema.csv', low_memory=False, header=[0,1])

freeform_df.columns = freeform_df.columns.map('_'.join)

multi_df.columns = multi_df.columns.map('_'.join)

SurveySchema.columns = SurveySchema.columns.map('_'.join)



data_scientist = multi_df['Q26_Do you consider yourself to be a data scientist?'] == "Definitely yes"

#datasci_df = multi_df[data_scientist]

top_earning_sal = {"100-125,000","125-150,000","150-200,000","200-250,000","250-300,000","300-400,000","400-500,000","500,000+"}

top_earn = multi_df['Q9_What is your current yearly compensation (approximate $USD)?'].isin(top_earning_sal)

top_dsci_df  = multi_df[data_scientist & top_earn]



age_df = multi_df['Q2_What is your age (# years)?'].value_counts()

age_df.index.name = 'Age'

age_df.sort_index(inplace= True)

age_df.plot(kind='bar',rot=20, title='Age distribution of Kagglers',figsize=(14,5));
recommendedlang_df = top_dsci_df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()

recommendedlang_df.index.name = 'Recommendation'

#reclang_teds_df.sort_index(inplace=True)

recommendedlang_df.plot(kind='bar',rot=0, title='Recommended Programming Language of Top Earning Data Science Kagglers',figsize=(10,5));
onlinelearn_teds_df = top_dsci_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()

onlinelearn_teds_df.index.name = 'MOOCS and Online Learning Platforms'

#onlinelearn_teds_df.sort_index(inplace=True)

onlinelearn_teds_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms the Top Earning Data Scientists Use',figsize=(15,5), colormap = 'Spectral');