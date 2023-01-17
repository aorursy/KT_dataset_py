# Import Python packages
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
from IPython.core import display as ICD
import warnings
%matplotlib inline
colors = ['aqua', 'fuchsia', 'red', 'lime', 'yellow', 'blue', 'orange', 'green', 'indigo', 'tomato', 'violet']
schema = pd.read_csv('../input/SurveySchema.csv')
# freeForm = pd.read_csv('../input/freeFormResponses.csv')
multChoice = pd.read_csv('../input/multipleChoiceResponses.csv')

# schema = pd.read_csv('SurveySchema.csv')
# # freeForm = pd.read_csv('freeFormResponses.csv')
# multChoice = pd.read_csv('multipleChoiceResponses.csv')

multChoiceData = multChoice[1:]
schema
multChoice.head()
multChoiceData['Q10'].value_counts().plot.bar(figsize = (10, 6), color = colors, 
    title ='Q10 Does your current employer incorporate machine learning methods into their business?', 
    fontsize=16)
noMLatWork = ['I do not know', 'No (we do not use ML methods)']
selfStarters = multChoiceData[multChoiceData['Q10'].isin(noMLatWork)]
len(selfStarters) / len(multChoiceData)
def make_subplots(question, plot_kind, title1, title2, figsize=(12,8)):
    fig, axarr = plt.subplots(2, 1, figsize=figsize)
    if plot_kind == 'barh':
        selfStarters[question].value_counts().plot(ax=axarr[0], kind='barh', color=colors, title=title1).invert_yaxis()
        axarr[0].tick_params(axis='x', labelbottom='on')
        multChoiceData[question].value_counts().plot(ax=axarr[1], kind='barh', color=colors, title=title2).invert_yaxis()
    elif plot_kind == 'bar':
        selfStarters[question].value_counts().sort_index().plot(ax=axarr[0], kind='bar', color=colors, title=title1)
        axarr[0].tick_params(axis='x', labelbottom='off')
        multChoiceData[question].value_counts().sort_index().plot(ax=axarr[1], kind='bar', color=colors, title=title2)
    elif plot_kind == 'pie':
        selfStarters[question].value_counts().plot(ax=axarr[0], kind=plot_kind, colors=colors, title=title1)
        multChoiceData[question].value_counts().plot(ax=axarr[1], kind=plot_kind, colors=colors, title=title2)   
    plt.subplots_adjust(hspace=.3)
make_subplots('Q1', 'barh', 'ML Self-starters - Gender', 'All respondents - Gender')
make_subplots('Q2', 'bar', 'ML Self-starters - Age', 'All respondents - Age')
make_subplots('Q4', 'barh', 'ML Self-starters - Education', 'All respondents - Education')
make_subplots('Q5', 'bar', 'ML Self-starters - Major', 'All respondents - Major')
make_subplots('Q6', 'barh', 'ML Self-starters - Current role', 'All respondents - Current role', figsize=(10, 16))
make_subplots('Q7', 'barh', 'ML Self-starters - Industry', 'All respondents - Industry', figsize=(10, 16))
make_subplots('Q8', 'bar', 'ML Self-starters - Experience (in years)', 'All respondents - Experience (in years)')
make_subplots('Q9', 'barh', 'ML Self-starters - Compensation', 'All respondents - Compensation', figsize=(10, 16))
make_subplots('Q17', 'pie', 'ML Self-starters - Programming language', 'All respondents - Programming language', figsize=(10, 22))
make_subplots('Q18', 'pie', 'ML Self-starters - Recommended programming language', 
              'All respondents - Recommended programming language', figsize=(10, 22))
make_subplots('Q23', 'barh', 'ML Self-starters - Time spent coding', 'All respondents - Time spent coding')
make_subplots('Q24', 'barh', 'ML Self-starters - Coding for data analysis experience', 
              'All respondents - Coding for data analysis experience')
make_subplots('Q25', 'barh', 'ML Self-starters - ML experience', 'All respondents - ML experience')
make_subplots('Q26', 'pie', 'ML Self-starters - Do you consider yourself to be a Data Scientist?', 
              'All respondents - Do you consider yourself to be a Data Scientist?', figsize=(10, 22))
make_subplots('Q32', 'bar', 'ML Self-starters - Type of data', 'All respondents - Type of data')
make_subplots('Q40', 'pie', 'ML Self-starters - Which better demonstrates expertise in data science?', 
              'All respondents - Which better demonstrates expertise in data science?', figsize=(10, 22))
make_subplots('Q43', 'bar', 'ML Self-starters - Exploring unfair bias (% of project)', 
              'All respondents - Exploring unfair bias (% of project)')
make_subplots('Q46', 'bar', 'ML Self-starters - Exploring model insights (% of project)', 
              'All respondents - Exploring model insights (% of project)')
make_subplots('Q48', 'pie', 'ML Self-starters - Do you consider ML models to be "black boxes"?', 
              'All respondents - Do you consider ML models to be "black boxes"?', figsize=(10, 22))