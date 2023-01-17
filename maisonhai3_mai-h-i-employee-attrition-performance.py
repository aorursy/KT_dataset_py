# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Libraries



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import iplot



import scipy

from scipy import stats

from sklearn.preprocessing import LabelEncoder



sns.set_palette('RdBu')
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print('Observations                                   : ', df.shape[0])

print('Features -- exclude Attrition and Satisfication: ', df.shape[1] - 2)
df.head(10)
df.tail(10)
df.columns
print('Nan data points: ', df.isnull().sum().sum())
df.Attrition.describe()
df.Attrition.value_counts()
df.JobSatisfaction.describe(percentiles=[0.01, 0.45, 0.90])
# The big picture

fig = make_subplots(rows=1, cols=2,

                   specs=[[{"type": "bar"}, {"type": "domain"}]])



# Sketch smaller details

trace0 = go.Histogram(x=df['Attrition'], name='In number', marker={'color':['red', 'blue']},

                     showlegend=False)

trace1 = go.Pie(values=df['Attrition'].value_counts(), name='Percentage', labels=['No', 'Yes'],

               textinfo='label+percent')



# Add traces

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



# Customize

fig.update(layout_title_text='<b> Attrition </b>')

fig.update_layout(showlegend=False)



# Done

fig.show()
# The big picture

fig = make_subplots(rows=3, cols=2,

                   specs=[[{'rowspan':3}, {"type": "domain"}],

                          [None,          {"type": "domain"}],

                          [None,          {"type": "domain"}]])



# Sketch smaller details



## The bar chart - with Yes = negative columns.

labels = ['R&D', 'Sales', 'HR']



yes = df['Department'][df.Attrition=='Yes'].value_counts()

trace_yes = go.Bar(x=labels, y=-yes, marker={'color':'red'}, showlegend=False) 



no  = df['Department'][df.Attrition=='No'].value_counts()

trace_no  = go.Bar(x=labels, y=no, marker={'color':'blue'}, showlegend=False )



## Pie 1 -- upper right

RD = df['Attrition'][df.Department=='Research & Development'].value_counts()

trace_3   = go.Pie(labels=['No', 'Yes'], values=RD, name='RD')



## Pie 2

Sales = df['Attrition'][df.Department=='Sales'].value_counts()

trace_4   = go.Pie(labels=['No', 'Yes'], values=Sales, name='Sales')



## Pie 3

HR = df['Attrition'][df.Department=='Human Resources'].value_counts()

trace_5   = go.Pie(labels=['No', 'Yes'], values=HR, name='HR')



# Add traces

fig.append_trace(trace_yes, 1, 1)

fig.append_trace(trace_no, 1, 1)



fig.append_trace(trace_3, 1, 2)

fig.append_trace(trace_4, 2, 2)

fig.append_trace(trace_5, 3, 2)



# Customize

fig.update(layout_title_text='<b> Attrition by Department </b>')



# Done

fig.show()
fig = px.box(df, y='MonthlyIncome', x='Gender', color='Gender', 

             points='all', 

             color_discrete_map={'Female':'red', 'Male':'Green'})



fig.update(layout_title_text='<b> Monthly Income by Gender </b>')

fig.update_layout(showlegend=False)



fig.show()
# The big picture

fig = make_subplots(rows=6, cols=2,

                   specs=[[{'rowspan':6}, {"type": "domain"}], # 1  --  1

                          [None,          {"type": "domain"}], # 0  --  2

                          [None,          {"type": "domain"}], # 0  --  3

                          [None,          {"type": "domain"}], # 0  --  4

                          [None,          {"type": "domain"}], # 0  --  5

                          [None,          {"type": "domain"}]])# 0  --  6



# Sketching

## Bar chart

labels=['Life Sciences', 'Medical','Marketing', 'Technical Degree', 'Other', 'Human Resources']



yes = df['EducationField'][df.Attrition=='Yes'].value_counts(ascending=False)

no = df['EducationField'][df.Attrition=='No'].value_counts(ascending=False)



fig.add_bar(y=-yes, x=labels, col=1, row=1, marker={'color':'red'},  showlegend=False)

fig.add_bar(y=no,   x=labels, col=1, row=1, marker={'color':'blue'}, showlegend=False)



## Pie chart

LS     = df['Attrition'][df.EducationField=='Life Sciences'].value_counts()

Med    = df['Attrition'][df.EducationField=='Medical'].value_counts()

Mar    = df['Attrition'][df.EducationField=='Marketing'].value_counts()

Tech   = df['Attrition'][df.EducationField=='Technical Degree'].value_counts()

Other  = df['Attrition'][df.EducationField=='Other'].value_counts()

HR     = df['Attrition'][df.EducationField=='Human Resources'].value_counts()



fig.add_pie(labels=['No', 'Yes'], values=LS,    name='LS',    col=2, row=1)

fig.add_pie(labels=['No', 'Yes'], values=Med,   name='Med',   col=2, row=2)

fig.add_pie(labels=['No', 'Yes'], values=Mar,   name='Mar',   col=2, row=3)

fig.add_pie(labels=['No', 'Yes'], values=Tech,  name='Tech',  col=2, row=4)

fig.add_pie(labels=['No', 'Yes'], values=Other, name='Other', col=2, row=5)

fig.add_pie(labels=['No', 'Yes'], values=HR,    name='HR',    col=2, row=6)



# Customize

fig.update(layout_title_text='<b> Attrition by Education Field </b>')

# Done

fig.show()
fig = make_subplots(rows=2, cols=2)



trace0 = go.Histogram(x=df['Department'], y=df['JobSatisfaction'], histfunc='avg')

trace1 = go.Histogram(x=df['EducationField'], y=df['JobSatisfaction'], histfunc='avg')

trace2 = go.Histogram(x=df['OverTime'], y=df['JobSatisfaction'], histfunc='avg')

trace3 = go.Histogram(x=df['MaritalStatus'], y=df['JobSatisfaction'], histfunc='avg')



fig.add_trace(trace0, 1, 1)

fig.add_trace(trace1, 1, 2)

fig.add_trace(trace2, 2, 1)

fig.add_trace(trace3, 2, 2)

#fig = px.histogram(df, x='Department', y='JobSatisfaction',  histfunc='avg')





fig.show()
g = sns.FacetGrid(data=df, row = 'Attrition', col = 'JobSatisfaction')

g.map(plt.hist, 'MonthlyIncome', bins=10)
sns.catplot(x='EducationField', y='MonthlyIncome',  data=df,

           kind='violin')

plt.xticks(rotation=45)
sns.countplot(x='DistanceFromHome', data=df)
# DistanceFromHome -- Attrition

sns.catplot(x='Attrition', y='DistanceFromHome', data=df,

           kind='box')
fig = px.histogram(df, x='JobSatisfaction', color='JobSatisfaction')



fig.update_layout(title='<b> JobSatisfaction </b>',

                  xaxis={'tickmode': 'array',

                         'tickvals': [1, 2, 3, 4]})



fig.show()
features_to_analysis =      ['OverTime',         'MonthlyIncome',         'Age',

                             'DistanceFromHome', 'TotalWorkingYears',     'MaritalStatus',

                             'JobLevel',         'NumCompaniesWorked',    'YearsSinceLastPromotion',

                             'MonthlyRate',      'TrainingTimesLastYear', 'YearsWithCurrManager',

                             'Education',        'PercentSalaryHike']

features_to_analysis.sort()

print(features_to_analysis)
# Create table of feature datatypes.

table_datatypes = pd.DataFrame(columns=['Features', 'Datatype'])



# 1st column: Features

table_datatypes['Features'] = features_to_analysis



# 2nd column: Datatypes

table_datatypes['Datatype'] = [df[feature].dtypes for feature in features_to_analysis]



print(table_datatypes)
# Binary encoding: MaritalStatus and OverTime:

lb = LabelEncoder()



df['MaritalStatus_encoded'] = lb.fit_transform(df['MaritalStatus']).astype(int)

df['OverTime_encoded'] = lb.fit_transform(df['OverTime']).astype(int)



# Origins replaced by encoded

features_to_analysis = ['MaritalStatus_encoded' if x=='MaritalStatus' else x for x in features_to_analysis]

features_to_analysis = ['OverTime_encoded' if x=='OverTime' else x for x in features_to_analysis]
# Split df to Yes-No Attrition

df_Attrition_yes = df[df.Attrition == 'Yes']

df_Attrition_no = df[df.Attrition == 'No']



# Run: One sample Two-sided T-test

t_statistic = []

p_value     = []



for feature in features_to_analysis:

    # t-test

    sample  = df_Attrition_yes[feature]

    popmean = df_Attrition_no[feature].mean() # mean of population

    t_stats, p = stats.ttest_1samp(sample, popmean)

           

    t_statistic.append(t_stats)

    p_value.append(p)    

    

    print('Feature: ', feature)

    print('t-statistic: %4.2f -- p-value: %4.4f \n' %(t_stats, p))
# Create tabel

table = pd.DataFrame()

table['Features'] = features_to_analysis

table['t-statistic'] = t_statistic

table['p-value'] = p_value



# Conclusions

alpha = 0.05

table['Decisions'] = ['Rejected' if x<alpha else 'Failed to reject' for x in table['p-value']]

table['Key factors'] = ['Yes' if x=='Rejected' else 'No' for x in table['Decisions']]



# Drop not-needed

#table = table.drop(['t-statistic', 'p-value'], axis=1)



print(table[['Features', 'Decisions', 'Key factors']].sort_values(by='Key factors', ascending=False))



#print(table.sort_values(by='Decisions'))
# Preparing samples for ANOVA

population = df[['MaritalStatus', 'OverTime', 'JobSatisfaction']]



anova_samples = {}

i = 1



# Create Samples by conditions

for MS in population['MaritalStatus'].unique():

    for OT in population['OverTime'].unique():

        sample = population['JobSatisfaction'][(df.MaritalStatus==MS) & (df.OverTime==OT)]

        sample.reset_index(drop=True, inplace=True)

        anova_samples[i] = sample

        

        i += 1
f, p = stats.f_oneway(anova_samples[1],

                      anova_samples[2],

                      anova_samples[3],

                      anova_samples[4],

                      anova_samples[5],

                      anova_samples[6])



print('F-statistic: %4.2f' %(f))

print('p-value    : %4.2f' %(p))
## Features to run correlation



# Those be analyzed already.

features_to_analysis.remove('MaritalStatus_encoded')

features_to_analysis.remove('OverTime_encoded')



# Put JobSatisfaction in to determine Correlation matrix latter

features_to_analysis.append('JobSatisfaction')



print(features_to_analysis)
corr_matrix = df[features_to_analysis].corr()



# The heatmap

figure = plt.figure(figsize=(16,12))



mask = np.triu(corr_matrix) # Hide the upper part.

sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cmap="YlGnBu", mask=mask)



plt.show()