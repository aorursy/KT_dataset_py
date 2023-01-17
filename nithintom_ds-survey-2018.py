import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.ticker import FuncFormatter
import warnings
warnings.simplefilter(action='ignore')
free = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/freeFormResponses.csv', sep=',', low_memory=False)
mc = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv', sep=',', low_memory=False) 
survey = pd.read_csv(r'/kaggle/input/kaggle-survey-2018/SurveySchema.csv', sep=',', low_memory=False)
print(f'freeFormResponses: {free.shape}')
print(f'freeFormResponses: {free.columns}')
print(f'multipleChoiceResponses: {mc.shape}')
print(f'multipleChoiceResponses: {mc.columns}')
print(f'SurveySchema: {survey.shape}')
print(f'SurveySchema: {survey.columns}')

qns = survey.loc[0].values.tolist()
qn_nos = survey.loc[0].index.tolist()
qns = dict(zip(qn_nos[1:], qns[1:]))
for n, q in qns.items():
    print(n, q)
    print()
categories = survey.iloc[:,0].values.tolist()
for i, q in enumerate(categories):
    print(i, q)
    print()
df = mc.copy()
df = df.drop([0])   # remove first row

# Start of time column
df['Time from Start to Finish (seconds)'] = df['Time from Start to Finish (seconds)'].apply(int)
# Rejecting those who answered questions too fast:
df = df[df['Time from Start to Finish (seconds)']>60]
df.drop(['Time from Start to Finish (seconds)'],axis=1,inplace=True)


print(f'Question: {qns["Q9"]}')
df['Q9'].unique()
def is_salary_known(x):
    if (x=='I do not wish to disclose my approximate yearly compensation' or x!=x): return 'no'
    return 'yes'

df['salary known'] = df['Q9'].apply(lambda x: is_salary_known(x))
all_salaries = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
                       '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
                       '200-250,000','250-300,000','300-400,000','400-500,000','500,000+']
sns.set_palette(['#95d0fc','pink'])
px = df['salary known'].value_counts(normalize=True).plot(kind='pie', autopct='%.1f%%', figsize=(16,7), fontsize=16)
px.set_title('Is salary info given',fontsize=20)
px.axis('equal')
px.get_yaxis().set_visible(False)
fig = plt.subplots(figsize=(16,7))
df_with_unknown_salaries = df
g1 = sns.countplot(x='salary known',data=df_with_unknown_salaries, 
                   order=['no','yes'],palette='pastel')
g1.set_xlabel('')
g1.set_ylabel('')
g1.tick_params(labelsize=16)
out = g1.set_title('Is salary info given',fontsize=18)
df = df[(df['salary known']=='yes')]
df = df[df.Q6 != 'Student']
df = df[df.Q7 != 'I am a student']
fig, ax = plt.subplots(figsize=(18,6))
g = sns.countplot(x='Q9',data=df, order=all_salaries, ax=ax,palette = 'Spectral')
g.set_xticklabels(ax.get_xticklabels(),rotation=90)
g.set_title('Yearly salary distribution', fontsize=20)
g.set_xlabel('Salary USD')
g.set_ylabel('Total Respondents')

# Add the percentage values above each bar
ncount = df.shape[0]
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),ha='center', va='bottom')
print(f'Question: {qns["Q3"]}')
df['Q3'].unique()
def rename_countries(x):
    if (x=='United States of America'): return 'USA'
    if (x=='United Kingdom of Great Britain and Northern Ireland'): return 'UK'
    return x

df['Q3'] = df['Q3'].apply(lambda x: rename_countries(x))

def extract_avg_pay(compensation):
    result = re.split('-|,',compensation)
    return 1000*(int(result[0]) + int(result[1]))/2
    
df = df[df.Q9 != '500,000+']
df['pay'] = df['Q9'].apply(extract_avg_pay)

sns.set_palette('Spectral')
px = df.groupby(['Q3'])['pay'].mean().sort_values(ascending=False).head(15).plot(kind="bar", figsize=(16,7))
px.set_title('Top mean salaries by country',fontsize=20)
px.set(xlabel='Country', ylabel='Mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=30)
sns.set_palette('husl')
px = df[df.Q3 != 'Other']['Q3'].value_counts().nlargest(15).plot(kind='bar', figsize=(16,7))
px.set_title('Respondents by country',fontsize=20)
_ = px.set_xticklabels(px.get_xticklabels(), rotation=30)
df_USA = df[(df['Q3']=='USA')]
df_India = df[(df['Q3']=='India')]
df_for_plot = pd.concat([df_USA,df_India])

fig, ax2 = plt.subplots(figsize=(18,6))
g2 = sns.countplot(x='Q9',data=df_for_plot, 
                   order=all_salaries, ax=ax2, hue='Q3')
g2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
g2.set_title('Yearly compensation distribution India vs USA',fontsize=20)
g2.set_ylabel('People in this range')
g2.set_xlabel('yearly compensation [USD]',fontsize=14)
g2.tick_params(labelsize=14)
plt.gca().legend().set_title('')
_ = plt.setp(ax2.get_legend().get_texts(), fontsize='17') # for legend text
print(f'Question: {qns["Q1"]}')
df['Q1'].unique()
sns.set_palette(['#95d0fc','pink'])
px = df[df.Q1.isin(['Female','Male'])]['Q1'].value_counts(normalize=True).plot(kind='pie', autopct='%.1f%%', figsize=(16,7))
px.set_title('Respondents by gender',fontsize=20)
px.axis('equal')
px.get_yaxis().set_visible(False)
px = df[df.Q1.isin(['Female','Male'])].groupby(['Q1'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set_title('Average pay by gender',fontsize=20)
px.set(xlabel='Gender', ylabel='mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=0)
print(f'Question: {qns["Q17"]}')
df['Q17'].unique()
sns.set_palette('pastel')
px = df.groupby(['Q17'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set_title('Average pay by programming language',fontsize=20)
px.set(xlabel='Language', ylabel='Mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=30)
print(f'Question: {qns["Q20"]}')
df['Q20'].unique()
px = df[df.Q20 != 'Other'].groupby(['Q20'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set_title('Average pay by ML library used',fontsize=20)
px.set(xlabel='Library', ylabel='Mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=30)
print(f'Question: {qns["Q22"]}')
df['Q22'].unique()
px = df[df.Q22 != 'Other'].groupby(['Q22'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set_title('Average pay by visualization library used',fontsize=20)
px.set(xlabel='Visualization', ylabel='Mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=30)
print(f'Question: {qns["Q8"]}')
df['Q8'].unique()
px = df.groupby(['Q8'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set_title('Average pay by experience',fontsize=20)
px.set(xlabel='Years of Experience', ylabel='Mean pay')
_ = px.set_xticklabels(px.get_xticklabels(), rotation=0)