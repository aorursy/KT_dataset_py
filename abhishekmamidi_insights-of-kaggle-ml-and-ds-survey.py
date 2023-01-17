import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from plotly import tools
import plotly.graph_objs as go
from IPython.core import display as ICD
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")
PATH = '../input/'
surveySchema = pd.read_csv(PATH + 'SurveySchema.csv')
freeFormResponses = pd.read_csv(PATH + 'freeFormResponses.csv')
multipleChoiceResponses = pd.read_csv(PATH + 'multipleChoiceResponses.csv', encoding='ISO-8859-1')
multipleChoiceResponses.head(2)
gender_distribution = multipleChoiceResponses['Q1'].iloc[1:].value_counts()
ax = gender_distribution.plot('bar', figsize=(10,6), width=0.3,
                                rot=0, title='Gender Distribution')
print('Number of respondents: ' + str(sum(gender_distribution)))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
age_order = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70-79', '80+']

age_distribution = multipleChoiceResponses['Q2'].iloc[1:].value_counts()
ax = age_distribution.loc[age_order].plot('bar', figsize=(15,6), rot=0, title='Age Distribution')
print('Number of respondents: ' + str(sum(age_distribution)))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
country_distribution = multipleChoiceResponses['Q3'].iloc[1:]
country_distribution = country_distribution.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom of \n Great Britain and \n Northern Ireland')
country_distribution = country_distribution.replace('United States of America', 'United States of \n America').value_counts()[:10]

ax = country_distribution.plot('bar', figsize=(15,6), rot=0, title='Country distribution')
print('Number of respondents: ' + str(sum(country_distribution)))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
type_of_education = ['Masterâs degree', 'Bachelorâs degree', 'Doctoral degree', 'Some college/university study without earning a bachelorâs degree', 'Professional degree', 'I prefer not to answer', 'No formal education past high school']
type_of_education_change = ['Masters \n degree', 'Bachelors \n degree', 'Doctoral \n degree', 'Some college/ \n university study \n without earning \n a bachelors degree', 'Professional \n degree', 'I prefer not \n to answer', 'No formal \n education past \n high school']

level_of_education = multipleChoiceResponses['Q4'].iloc[1:].value_counts()

for i in range(len(type_of_education)):
    level_of_education = level_of_education.replace(type_of_education[i], type_of_education_change[i])

ax = level_of_education.plot('bar', figsize=(15,6),rot=0, title='Distribution of level of formal education')
print('Number of respondents: ' + str(sum(level_of_education)))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
undergraduate_major = multipleChoiceResponses['Q5'].iloc[1:].value_counts()

ax = undergraduate_major.plot('barh', figsize=(15,10),rot=0, fontsize=13,
                                title='Distribution of Undergraduate major field')
print('Number of respondents: ' + str(sum(undergraduate_major)))
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.4, \
            str(round(i.get_width(), 2)), fontsize=15, color='black')
ax.invert_yaxis()
current_role = multipleChoiceResponses['Q6'].iloc[1:].value_counts()

ax = current_role.plot('barh', figsize=(15,10),rot=0, fontsize=13,
                      title='Distribution of Current role')
print('Number of respondents: ' + str(sum(current_role)))
for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.4, \
            str(round(i.get_width(), 2)), fontsize=15, color='black')
ax.invert_yaxis()
years_of_experience = multipleChoiceResponses['Q8'].iloc[1:]

ax = years_of_experience.value_counts().plot('bar', figsize=(15,6),rot=0, fontsize=13,
                                            title='Years of Experience')
print('Number of respondents: ' + str(sum(years_of_experience.value_counts())))
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
compensation = multipleChoiceResponses['Q9'].iloc[1:]
compensation = compensation.replace(multipleChoiceResponses['Q9'].iloc[1:][6], 
                     'I do not wish to disclose \n my yearly compensation')

ax = compensation.value_counts().plot('barh', figsize=(15,10),rot=0, fontsize=13,
                                            title='Current compensation')
print('Number of respondents: ' + str(sum(compensation.value_counts())))

for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.4, \
            str(round(i.get_width(), 2)), fontsize=15, color='black')
ax.invert_yaxis()
type_of_usage = ['We are exploring ML methods (and may one day put a model into production)',
                'No (we do not use ML methods)',                                         
                'We recently started using ML methods (i.e., models in production for less than 2 years)',
                'We have well established ML methods (i.e., models in production for more than 2 years)',
                'We use ML methods for generating insights (but do not put working models into production)']

type_of_usage_change = ['We are exploring ML methods \n (and may one day put a model into production)',
                'No (we do not use ML methods)',                                         
                'We recently started using ML methods \n (i.e., models in production for less than 2 years)',
                'We have well established ML methods \n (i.e., models in production for more than 2 years)',
                'We use ML methods for generating insights \n (but do not put working models into production)']

machine_learning_usage = multipleChoiceResponses['Q10'].iloc[1:]
for i in range(len(type_of_usage)):
    machine_learning_usage = machine_learning_usage.replace(type_of_usage[i], type_of_usage_change[i])

ax = machine_learning_usage.value_counts().plot('barh', figsize=(15,8),rot=0, fontsize=15,
                                            title='Machine Learning usage in current employment')
print('Number of respondents: ' + str(sum(years_of_experience.value_counts())))

for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.4, \
            str(round(i.get_width(), 2)), fontsize=15, color='black')
ax.invert_yaxis()
programming_language = multipleChoiceResponses['Q17'].iloc[1:]

ax = programming_language.value_counts()[:8].plot('bar', figsize=(15,6),rot=0, fontsize=13,
                                            title='Top Programming language')
print('Number of respondents: ' + str(sum(programming_language.value_counts())))

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
programming_language_recommendation = multipleChoiceResponses['Q18'].iloc[1:]

ax = programming_language_recommendation.value_counts()[:].plot('bar', figsize=(15,6),rot=0, fontsize=13,
                                            title='Recommendation of Programming language')
print('Number of respondents: ' + str(sum(programming_language.value_counts())))

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
                textcoords='offset points', fontsize=14, color='black')
ml_library_preference = multipleChoiceResponses['Q20'].iloc[1:].value_counts()
labels = (np.array(ml_library_preference.index))
sizes = (np.array((ml_library_preference)))
print('Number of respondents: ' + str(sum(ml_library_preference)))

trace = go.Pie(labels=labels, values=sizes, hole=0.6)
layout = go.Layout(
    title='Preference of ML library'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="ml-library")
plotting_library_preference = multipleChoiceResponses['Q22'].iloc[1:].value_counts()
labels = (np.array(plotting_library_preference.index))
sizes = (np.array((plotting_library_preference)))
print('Number of respondents: ' + str(sum(plotting_library_preference)))

trace = go.Pie(labels=labels, values=sizes, hole=0.6)
layout = go.Layout(
    title='Preference of plotting library'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
time_devotion = multipleChoiceResponses['Q23'].iloc[1:]
ax = time_devotion.value_counts()[:].plot('barh', figsize=(15,6),rot=0, fontsize=13,
                                            title='Time devotion for ML/DS')
print('Number of respondents: ' + str(sum(time_devotion.value_counts())))

for i in ax.patches:
    ax.text(i.get_width()+50, i.get_y()+.4, \
            str(round(i.get_width(), 2)), fontsize=15, color='black')
ax.invert_yaxis()
how_long = multipleChoiceResponses['Q24'].iloc[1:].value_counts()
labels = (np.array(how_long.index))
sizes = (np.array((how_long)))
print('Number of respondents: ' + str(sum(how_long)))

trace = go.Pie(labels=labels, values=sizes, hole=0.6)
layout = go.Layout(
    title='How long have they been writing code to analyze data?'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="How long have they been writing code to analyze data?")
percentage_of_ML = multipleChoiceResponses['Q46'].iloc[1:].value_counts()
labels = (np.array(percentage_of_ML.index))
sizes = (np.array((percentage_of_ML)))
print('Number of respondents: ' + str(sum(percentage_of_ML)))

trace = go.Pie(labels=labels, values=sizes, hole=0.6)
layout = go.Layout(
    title='percentage of Machine learning/Data Science is used in the projects'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="percentage of Machine learning/Data Science is used in the projects")