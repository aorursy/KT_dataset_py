import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

sns.set_palette('Paired')



import os

from os.path import join



path = "../input/stack-overflow-2018-developer-survey/"
df = pd.read_csv(join(path, 'survey_results_public.csv'), low_memory=False)
df.shape
df.head()
pd.options.display.max_colwidth = 400

schema = pd.read_csv(join(path, 'survey_results_schema.csv'), low_memory=False)
def plot_categorical_count(df, column, title='', limit=2, xtick_rotation='horizontal'):

    column_count = df[column].value_counts()[:limit]

    

    fig = plt.figure(figsize=(14, 8))

    sns.barplot(x=column_count.index, y=column_count.values, palette='Paired')

    sns.despine(left=True)

    plt.title(title, fontsize=16)

    plt.xticks(rotation=xtick_rotation)



def plot_heatmap(df, variable):

    l = []

    for name, group in top10_df.groupby('Country'):

        dff = pd.DataFrame(group[variable].value_counts() / group[variable].count()) 

        dff['Country'] = name

        dff['rate'] = dff.index

        l.append(dff)

    fig = plt.figure(figsize=(14, 8))

    df_2 = pd.concat(l)

    dff = df_2.pivot_table(index='Country', columns='rate')

    sns.heatmap(dff, cmap="YlGnBu", linewidths=.3)
plot_categorical_count(df, column='Country', title='Number of respondant per country', limit=10)
plot_categorical_count(df, 'Hobby', title='Number of respondent who code as a hobby vs those who don\'t')
plot_categorical_count(df, 'OpenSource', title='Number of Open Source Contributor')

open_source_count = df['OpenSource'].value_counts()



plt.figure(figsize=(14, 8))

sns.countplot(df['Hobby'], palette='Paired', hue=df['OpenSource'])

sns.despine(left=True)

plt.title('Number of Open Source Contributour', fontsize=16)
plot_categorical_count(df, 'Student', title='Number Of Students', limit=3)
plot_categorical_count(df, 'Employment', title='Employement Statuses of respondants', limit=6, xtick_rotation='vertical')
plot_categorical_count(df, 'Gender', title='Number of respondents per Gender')
plot_categorical_count(df, 'UndergradMajor', 'Majors count', limit=100, xtick_rotation='vertical')
top_10_list = list(df['Country'].value_counts()[:10].index)

def st_not(row):

    if 'dissatisfied' in row:

        return 'Dissatisfied'

    return 'Satisfied'



df['sat_or_not'] = df['JobSatisfaction'].dropna().map(st_not)

top10_df = df.where(df['Country'].isin(top_10_list))

sat_count = pd.DataFrame()

names = []

sat = []

disat = []

for name, group in top10_df.groupby('Country'):

    names.append(name)

    country_count = group['sat_or_not'].value_counts()

    sat.append(country_count['Satisfied'])

sat_count['Country'] = names

sat_count
df['JobSatisfaction'].value_counts()

sat = df[np.logical_or(np.logical_or(df['JobSatisfaction'] == 'Moderately satisfied', df['JobSatisfaction'] == 'Extremely satisfied'), df['JobSatisfaction'] == 'Slightly satisfied')]



plt.figure(figsize=(14, 8))

sns.countplot(data=sat, x='Country', hue='JobSatisfaction', palette='Paired', order=sat['Country'].value_counts()[:10].index)

sns.despine(left=True)

plt.xticks(rotation='vertical')
df['CareerSatisfaction'].value_counts()

sat = df[np.logical_or(np.logical_or(df['CareerSatisfaction'] == 'Moderately satisfied', df['CareerSatisfaction'] == 'Extremely satisfied'), df['CareerSatisfaction'] == 'Slightly satisfied')]



plt.figure(figsize=(14, 8))

sns.countplot(data=sat, x='Country', hue='CareerSatisfaction', palette='Paired', order=sat['Country'].value_counts()[:10].index)

sns.despine(left=True)

plt.xticks(rotation='vertical')
plt.figure(figsize=(14, 12))

df_top10 = df.where(df['Country'].isin(top_10_list))



sns.boxplot(data=df_top10, x='ConvertedSalary', y='Country', palette='Paired')

plt.title('Salary Distribution in the Top 10 Countries', fontsize=16)

sns.despine(left=True)
fig, axes = plt.subplots(10, 1, figsize=(14, 34))



for ax, country in zip(axes, top_10_list):

    data = df[df['Country'] == country]

    sns.countplot(data=data, x='TimeFullyProductive', palette='Paired', ax=ax, order=data['TimeFullyProductive'].value_counts().index)

    ax.set_title('Productivity time in {}'.format(country), fontsize=16)

    sns.despine(left=True)

plt.subplots_adjust(hspace=.6)
fig, axes = plt.subplots(10, 1, figsize=(14, 34))



for ax, country in zip(axes, top_10_list):

    data = df[df['Country'] == country]

    sns.countplot(data=data, x='YearsCodingProf', palette='Paired', ax=ax, order=data['YearsCodingProf'].value_counts().index)

    ax.set_title('Years coding professionally in {}'.format(country), fontsize=16)

    sns.despine(left=True)

plt.subplots_adjust(hspace=.6)
sat = df

plt.figure(figsize=(14, 8))

sns.countplot(data=sat, x='Country', hue='HoursComputer', palette='Paired', order=sat['Country'].value_counts()[:10].index)

sns.despine(left=True)

plt.xticks(rotation='vertical')
sat = df

plt.figure(figsize=(14, 8))

sns.countplot(data=sat, x='Country', hue='HoursOutside', palette='Paired', order=sat['Country'].value_counts()[:10].index)

sns.despine(left=True)

plt.xticks(rotation='vertical')
plt.figure(figsize=(14, 8))

sns.countplot(data=df, x = 'Country', hue='SkipMeals', palette='Paired', order=df['Country'].value_counts()[:10].index)

sns.despine(left=True)
plt.figure(figsize=(14, 8))

sns.countplot(data=df, x='Country', hue='Exercise', palette='Paired', order=df['Country'].value_counts()[:10].index)

sns.despine(left=True)