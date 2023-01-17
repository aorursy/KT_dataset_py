# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import math
import sklearn
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
import itertools
multichoice = pd.read_csv('../input/multipleChoiceResponses.csv')
freeform = pd.read_csv('../input/freeFormResponses.csv')
schema = pd.read_csv('../input/SurveySchema.csv')
tempframe = pd.DataFrame()
pd.options.display.max_colwidth = 1550
tempframe["Number"] = list(schema)
tempframe["Question"] = schema.iloc[0].values
tempframe
tempframe.loc[(tempframe.Number == 'Q1') | (tempframe.Number == 'Q3') | (tempframe.Number == 'Q9')]
multichoice_processed = pd.DataFrame()
multichoice_processed[['Country',  'Gender', 'Salary']] = multichoice[['Q3', 'Q1', 'Q9']]
multichoice_processed
multichoice_processed = multichoice_processed.drop([0])
multichoice_processed
multichoice_processed.Country.value_counts()
multichoice_processed.Gender.value_counts()
multichoice_processed.Salary.value_counts()
multichoice_processed = multichoice_processed.dropna(how='any')
multichoice_processed = multichoice_processed[(multichoice_processed.Gender != 'Prefer not to say') & (multichoice_processed.Gender != 'Prefer to self-describe') 
& (multichoice_processed.Salary != 'I do not wish to disclose my approximate yearly compensation') & 
(multichoice_processed.Country != 'I do not wish to disclose my location') & 
(multichoice_processed.Country != 'Other')]
multichoice_processed.Country.value_counts()
multichoice_processed.Gender.value_counts()
multichoice_processed.Salary.value_counts()
multichoice_processed['Salary_Processed'] = multichoice_processed['Salary'].str.split('-').str[-1]
multichoice_processed['Salary_Processed'] = multichoice_processed['Salary_Processed'].str.split(',').str[0]
multichoice_processed['Salary_Processed'] = pd.to_numeric(multichoice_processed['Salary_Processed'])
multichoice_processed['Salary_Processed'] = multichoice_processed.Salary_Processed.astype(int)
multichoice_processed
multichoice_processed.Salary_Processed.describe()

median = multichoice_processed.Salary_Processed.median()
median
## Creating a new dataframe with the columns of interest and a median value

global_statistics = pd.DataFrame()
global_statistics = multichoice_processed.groupby('Country')['Salary_Processed'].median().reset_index().rename(columns={'Country': 'Country', 'Salary_Processed' : 'Country_Median'})

## Using temporary male/female summaries and mapping median values into the new dataframe

temp_male = pd.DataFrame()
temp_male = multichoice_processed[multichoice_processed['Gender']=='Male']
temp_male = temp_male.groupby(['Country'])['Salary_Processed'].median().reset_index().rename(columns={'Country': 'Country', 'Salary_Processed' : 'Male_Median'})
temp_male
temp_female = pd.DataFrame()
temp_female = multichoice_processed[multichoice_processed['Gender']=='Female']
temp_female = temp_female.groupby(['Country'])['Salary_Processed'].median().reset_index().rename(columns={'Country': 'Country', 'Salary_Processed' : 'Female_Median'})
temp_female
global_statistics = pd.merge(global_statistics, temp_male)
global_statistics = pd.merge(global_statistics, temp_female)

## Calculating the median difference

global_statistics['Median_Gap'] = global_statistics["Male_Median"] - global_statistics["Female_Median"]

## Defining and applying a classifier function based on the Country_Median column and a median variable

def process(row, median=''):
   if row['Country_Median'] < median:
      return 'Low-income'
   else:
       return 'High-income'
    
global_statistics['Classifier'] = global_statistics.apply(lambda row: process(row, median = median),axis=1)

global_statistics
    


data = [dict(
        type = 'choropleth',
        locations = global_statistics['Country'],
        locationmode = 'country names',
        z = global_statistics['Country_Median'],
        colorscale =[[0.0, 'rgb(84,39,143)'], [0.1, 'rgb(117,107,177)'], [0.2, 'rgb(158,154,200)'],
       [0.3, 'rgb(188,189,220)'], [0.4, '218,218,235)'], [0.5, 'rgb(240,240,240)'],
       [0.6, 'rgb(255,214,151)'],[0.8, 'rgb(250,195,104)'], [0.9, 'rgb(250,177,58)'],
       [1.0, 'rgb(252,153,6)']],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 0.5
            ) 
        ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Average income in 000s')
)
]

layout = dict(
    title = 'Global Income Distribution for Data Scientists and Software Engineers',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator')
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


countries = global_statistics.sort_values(by=['Country_Median']).Country.to_frame()
incomes = global_statistics.sort_values(by=['Country_Median']).Country_Median.to_frame()
countries_incomes = pd.merge(incomes, countries, left_index=True, right_index=True)
countries_incomes

plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)
plt.rcParams["figure.figsize"] = [30,10]

ax = countries_incomes.plot(x="Country", y="Country_Median", kind="bar", color = "blue")
ax.set_ylabel('Country', fontsize = '25')
ax.set_xlabel('Income', fontsize = '25')
ax.set_title('Global Median Incomes', fontsize = '25')
plt.show()
temp_high = global_statistics[global_statistics['Classifier']=='High-income']
temp_low = global_statistics[global_statistics['Classifier']=='Low-income']
temp_high.describe()
temp_low.describe()
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rcParams["figure.figsize"] = [30,10]

ax = global_statistics.plot(x="Country", y="Male_Median", kind="bar", color = "red")
global_statistics.plot(x="Country", y="Female_Median", kind="bar", ax=ax, color="blue")
ax.set_ylabel('Median Income', fontsize = '25')
ax.set_xlabel('Country', fontsize = '25')
ax.set_title('Male vs Female Median Incomes', fontsize = '25')
plt.show()

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rcParams["figure.figsize"] = [30,10]

ax = global_statistics.plot(x="Country", y=["Country_Median", "Median_Gap"], kind="bar")
ax.set_ylabel('Median Income / Median Gap', fontsize = '25')
ax.set_xlabel('Country', fontsize = '25')
ax.set_title('Median Income vs Median Gender Gap', fontsize = '25')
plt.show()
global_statistics["Median_Gap"].describe()
global_statistics["Normalized_Gap"] = global_statistics["Median_Gap"] / global_statistics["Country_Median"]
global_statistics
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.rcParams["figure.figsize"] = [20,10]

ax = global_statistics.plot(x="Country", y=["Normalized_Gap"], kind="bar", color="red")
ax.set_ylabel('Median Normalized Gap', fontsize = '25')
ax.set_xlabel('Country', fontsize = '25')
ax.set_title('Median Normalized Gender Gap', fontsize = '25')
ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show()
global_statistics["Normalized_Gap"].describe()
data = [dict(
        type = 'choropleth',
        locations = global_statistics['Country'],
        locationmode = 'country names',
        z = global_statistics['Median_Gap'],
        colorscale =[[0.0, 'rgb(84,39,143)'], [0.1, 'rgb(117,107,177)'], [0.2, 'rgb(158,154,200)'],
       [0.3, 'rgb(188,189,220)'], [0.4, '218,218,235)'], [0.5, 'rgb(240,240,240)'],
       [0.6, 'rgb(255,214,151)'],[0.8, 'rgb(250,195,104)'], [0.9, 'rgb(250,177,58)'],
       [1.0, 'rgb(252,153,6)']],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 0.5
            ) 
        ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Gap in 000s')
)
]

layout = dict(
    title = 'Global Distribution of the Gender Income Gap for Data Scientists and Software Engineers',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator')
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
data = [dict(
        type = 'choropleth',
        locations = global_statistics['Country'],
        locationmode = 'country names',
        z = global_statistics['Normalized_Gap'],
        colorscale =[[0.0, 'rgb(84,39,143)'], [0.1, 'rgb(117,107,177)'], [0.2, 'rgb(158,154,200)'],
       [0.3, 'rgb(188,189,220)'], [0.4, '218,218,235)'], [0.5, 'rgb(240,240,240)'],
       [0.6, 'rgb(255,214,151)'],[0.8, 'rgb(250,195,104)'], [0.9, 'rgb(250,177,58)'],
       [1.0, 'rgb(252,153,6)']],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 0.5
            ) 
        ),
        colorbar = dict(
            autotick = False,
            title = 'Gap in 100%')
)
]

layout = dict(
    title = 'Global Distribution of the Gender Income Gap for Data Scientists and Software Engineers',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator')
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
temp_high = global_statistics[global_statistics['Classifier']=='High-income']
temp_low = global_statistics[global_statistics['Classifier']=='Low-income']

ax.set_xlabel('Index', fontsize = '25')
ax.set_ylabel('Median Gap', fontsize = '25')
ax = temp_high.reset_index().sort_values(by=['Normalized_Gap']).plot(kind='scatter', x='index', y='Median_Gap',
                                           color='Red', label='High Income Gap')

ax.set_xlabel('Index', fontsize = '25')
ax.set_ylabel('Median Gap', fontsize = '25')
temp_low.reset_index().sort_values(by=['Normalized_Gap']).plot(kind='scatter', x='index', y='Median_Gap',
                                          color='Blue', label='Low Income Gap', ax=ax)
temp_high = global_statistics[global_statistics['Classifier']=='High-income']
temp_low = global_statistics[global_statistics['Classifier']=='Low-income']

ax.set_xlabel('Index', fontsize = '25')
ax.set_ylabel('Median Gap', fontsize = '25')
ax = temp_high.reset_index().sort_values(by=['Normalized_Gap']).plot(kind='scatter', x='index', y='Normalized_Gap',
                                           color='Red', label='High Income Gap')

ax.set_xlabel('Index', fontsize = '25')
ax.set_ylabel('Median Gap', fontsize = '25')
temp_low.reset_index().sort_values(by=['Normalized_Gap']).plot(kind='scatter', x='index', y='Normalized_Gap',
                                          color='Blue', label='Low Income Gap', ax=ax)
temp_high.describe()
temp_low.describe()
tempframe.loc[(tempframe.Number == 'Q4')]
multichoice_processed = pd.DataFrame()
multichoice_processed[['Country',  'Gender', 'Salary', 'Education']] = multichoice[['Q3', 'Q1', 'Q9', 'Q4']]
multichoice_processed = multichoice_processed.drop([0])
multichoice_processed = multichoice_processed.dropna(how='any')
multichoice_processed = multichoice_processed[(multichoice_processed.Gender != 'Prefer not to say') & (multichoice_processed.Gender != 'Prefer to self-describe') 
& (multichoice_processed.Salary != 'I do not wish to disclose my approximate yearly compensation') & 
(multichoice_processed.Country != 'I do not wish to disclose my location') & 
(multichoice_processed.Country != 'Other') & (multichoice_processed.Education != 'I prefer not to answer')]
multichoice_processed
multichoice_processed['Education'].value_counts()
multichoice_processed['Classifier'] = multichoice_processed.Country.map(global_statistics.set_index('Country')['Classifier'])
multichoice_processed
male_education = multichoice_processed[multichoice_processed['Gender'] == 'Male']
male_education['Education'].value_counts()
female_education = multichoice_processed[multichoice_processed['Gender'] == 'Female']
female_education['Education'].value_counts()
pd.value_counts(male_education['Education']).plot.bar(title='Highest Level of Formal Education in Males', fontsize='15')

pd.value_counts(female_education['Education']).plot.bar(title='Highest Level of Formal Education in Females', fontsize='15')
hi_education = multichoice_processed[multichoice_processed['Classifier'] == 'High-income']
hi_education['Classifier'].value_counts()
lo_education = multichoice_processed[multichoice_processed['Classifier'] == 'Low-income']
lo_education['Classifier'].value_counts()

pd.value_counts(hi_education['Education']).plot.bar(title='Highest Level of Formal Education in High Income Countries', fontsize='15')
pd.value_counts(lo_education['Education']).plot.bar(title='Highest Level of Formal Education in High Income Countries', fontsize='15')