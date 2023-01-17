# importing packages

import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.options.display.max_columns = 9999
from matplotlib import rc

import warnings
warnings.simplefilter("ignore")


%matplotlib inline

# opening data

df = pd.read_csv(r'../input/multipleChoiceResponses.csv',header=1)

cols_keep = ['Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
             'What is your age (# years)?',
             'What is your current yearly compensation (approximate $USD)?',
             'In which country do you currently reside?',
             'Does your current employer incorporate machine learning methods into their business?',
             'What is your gender? - Selected Choice',
             'How long have you been writing code to analyze data?',
             'How many years of experience do you have in your current role?']

cols_programming = [col for col in df.columns if 'What programming languages do you use on a regular basis' in col]
cols_programming.remove('What programming languages do you use on a regular basis? (Select all that apply) - Other - Text')

cols_keep.extend(cols_programming)

dict_change = {'What is your age (# years)?':'age',
               'What is your current yearly compensation (approximate $USD)?':'compensation',
               'In which country do you currently reside?':'country',
               'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'role',
               'Does your current employer incorporate machine learning methods into their business?':'use_ml',
               'What is your gender? - Selected Choice':'gender',
               'How long have you been writing code to analyze data?':'time_coding',
               'How many years of experience do you have in your current role?':'time_work'}

dict_programming = {}
for i in cols_programming:
    dict_programming[i] = i.split('-')[2]
    
    
dict_change.update(dict_programming)
                   
df2 = df[cols_keep]
                   
df2 = df2.rename(columns=dict_change)

for i in dict_programming.values():
    df2[i] = df2[i].fillna(0)
    df2[i] = np.where(df2[i] == 0, 0 ,1)
    df2[i] = df2[i].astype(int)
    
df2['Number_languages'] = df2[list(dict_programming.values())].sum(1)
                   
#df2['time_work'] = df2['time_work'].fillna('0-1')
df2['time_work'] = df2['time_work'].replace('30 +','25-30')
df2['time_work'] = df2['time_work'].replace('25-30','10-15')
df2['time_work'] = df2['time_work'].replace('20-25','10-15')
df2['time_work'] = df2['time_work'].replace('15-20','10-15')

df2['time_work'] = df2['time_work'].str.split('-', expand = True)
df2['time_work'] = df2['time_work'].astype(float)
                   
df2['age'] = df2['age'].replace('80+','70-79')
df2['age'] = df2['age'].str.split('-', expand = True)
df2['age'] = df2['age'].astype(int)
df2 = df2[df2['compensation'].notnull()]
df2 = df2[df2['compensation'] != 'I do not wish to disclose my approximate yearly compensation']
df2['compensation'] =  df2['compensation'].str.split('-').apply(pd.Series)[0]
df2['compensation'] = df2['compensation'].replace('500,000+',500)
df2['compensation'] = df2['compensation'].astype(int)
ax, fig = plt.subplots(1,1, figsize= (16,6))
ax = sns.countplot(y="role",
                   data=df2,
                   order = df2.groupby(['role'])['country'].count().sort_values(ascending=False).index)
df3 = df2[df2['role'] == 'Data Scientist']
fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="time_work",
            orient='h',
            data=df3,
            edgecolor='black',
            linewidth=1.5,
            ax=ax[1])

df4 = df3.melt(value_vars=dict_programming.values())
ax, fig = plt.subplots(1,1, figsize= (16,6))
ax = sns.barplot(x="value",y='variable',
                 order = df4.groupby(['variable'])['value'].sum().sort_values(ascending=False).index,
                 data=df4)#,
fig, ax = plt.subplots(1,1, figsize= (18,6))

ax.set_title('Number of languages know by Data Scientists')

sns.countplot(x="Number_languages",
              orient='h',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax)
df3['Number_languages'] = np.where(df3['Number_languages'] > 6, 6, df3['Number_languages'])
fig, ax = plt.subplots(1,2, figsize= (18,8))
sns.boxplot(y="Number_languages",x='compensation',
            orient='h',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="Number_languages",
            orient='h',
            data=df3,
            edgecolor='black',
            linewidth=1.5,
            ax=ax[1])
df3['Python & R'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1), 1,0)
df3['Python & R & SQL'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1) & (df3[' SQL'] == 1), 1, 0)
df3['Python & SQL'] = np.where((df3[' SQL'] == 1) & (df3[' Python'] == 1), 1, 0)
df3['R & SQL'] = np.where((df3[' R'] == 1) & (df3[' Python'] == 1), 1, 0)

df3['Top_languages'] = np.where((df3[' R'] == 1) | (df3[' Python'] == 1) | (df3[' SQL'] == 1), 1, 0)
fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Top_languages',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience')
ax[1].set_title('Distribution of Years Experience')

sns.countplot(y="time_work",
              orient='h',
              hue='Top_languages',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' Python',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python')
ax[1].set_title('Distribution of Years Experience - Python')

sns.countplot(y="time_work",
              orient='h',
              hue=' Python',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' R',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - R')
ax[1].set_title('Distribution of Years Experience - R')

sns.countplot(y="time_work",
              orient='h',
              hue=' R',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue=' SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - SQL')
ax[1].set_title('Distribution of Years Experience - SQL')

sns.countplot(y="time_work",
              orient='h',
              hue=' SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Python & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python & SQL')
ax[1].set_title('Distribution of Years Experience - Python & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='Python & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='R & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - R & SQL')
ax[1].set_title('Distribution of Years Experience - R & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='R & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

fig, ax = plt.subplots(1,2, figsize= (18,6))
sns.boxplot(y="time_work",x='compensation',
            orient='h',
            hue='Python & R & SQL',
            data=df3,
            ax=ax[0])

ax[0].set_title('Compensation x Years Experience - Python & R & SQL')
ax[1].set_title('Distribution of Years Experience - Python & R & SQL')

sns.countplot(y="time_work",
              orient='h',
              hue='Python & R & SQL',
              data=df3,
              edgecolor='black',
              linewidth=1.5,
              ax=ax[1])

