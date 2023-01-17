# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-BCG_world_atlas_data-bcg_strain-7July2020.csv', encoding='utf8')

df.head()
plt.figure(figsize=(24, 6))

plt.subplot(131)

sns.countplot(df['bcg_strain_original'],order = df['bcg_strain_original'].value_counts(dropna=False).index)

plt.xticks(rotation=90)

plt.subplot(132)

sns.countplot(df['bcg_strain_du_class'],order = df['bcg_strain_du_class'].value_counts(dropna=False).index)

plt.subplot(133)

sns.countplot(df['bcg_strain_t_cell_grp_3'],order = df['bcg_strain_t_cell_grp_3'].value_counts(dropna=False).index)
fig, ax = plt.subplots(1,3, figsize = (20,6), sharex=True)

sns.countplot(x='bcg_strain_original',data=df, palette="copper", ax=ax[0])

sns.countplot(x='bcg_strain_original',hue='bcg_policy_first_year_original', palette="ocean", data=df,ax=ax[1])

sns.countplot(x='bcg_strain_original',hue='bcg_policy_last_year_original', palette="cubehelix", data=df,ax=ax[2])

ax[0].title.set_text('BCG Strain Original Count')

ax[1].title.set_text('BCG Strain Original Vs First Year Policy')

ax[2].title.set_text('BCG Strain Original Vs Last Year Policy ')

plt.show()
ax = df['bcg_policy_first_year_original'].value_counts().plot.barh(figsize=(16, 8))

ax.set_title('BCG Policies 1st Year Original', size=18)

ax.set_ylabel('bcg_policy_first_year_original', size=10)

ax.set_xlabel('bcg_strain_original', size=10)
ax = df['bcg_policy_last_year_original'].value_counts().plot.barh(figsize=(16, 8), color='r')

ax.set_title('BCG Policies Last Year Original', size=18)

ax.set_ylabel('bcg_policy_last_year_original', size=10)

ax.set_xlabel('bcg_strain_original', size=10)
plt.figure(figsize=(15, 10))

ax=sns.countplot(x="bcg_strain_original", data=df,palette="GnBu_d",edgecolor="black", order = df['bcg_strain_original'].value_counts().index)

plt.xticks(rotation=90)

plt.title('BCG Strain Original')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
plt.figure(figsize=(8, 6))

ax=sns.countplot(x="bcg_strain_du_class", data=df,palette="GnBu_d",edgecolor="black", order = df['bcg_strain_du_class'].value_counts().index)

plt.xticks(rotation=90)

plt.title('BCG Strain Du Class')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
plt.figure(figsize=(8, 6))

ax=sns.countplot(x="bcg_strain_t_cell_grp_3", data=df,palette="GnBu_d",edgecolor="black", order = df['bcg_strain_t_cell_grp_3'].value_counts().index)

plt.xticks(rotation=90)

plt.title('BCG Strain T Cell Group 3')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',

                    color= 'black')

plt.show()
fig = px.bar(df[['bcg_strain_original','country_name']].sort_values('country_name', ascending=False), 

                        y = "country_name", x= "bcg_strain_original", color='country_name', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="BCG Strain Original by Country")



fig.show()
fig = px.bar(df, 

             x='vaccination_timing', y='country_name', color_discrete_sequence=['#27F1E7'],

             title='BCG Vaccination Timing by Country', text='bcg_strain_original')

fig.show()
fig = px.bar(df[['bcg_policy_first_year_original','country_name']].sort_values('country_name', ascending=False), 

                        y = "country_name", x= "bcg_policy_first_year_original", color='country_name', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="BCG Policy 1st Year Original")



fig.show()
fig = px.bar(df[['bcg_policy_last_year_original','country_name']].sort_values('country_name', ascending=False), 

                        y = "country_name", x= "bcg_policy_last_year_original", color='country_name', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='forestgreen', size=14))

fig.update_layout(title_text="BCG Strain Last Year Original ")



fig.show()
sns.countplot(x="is_bcg_mandatory_for_all_children",data=df,palette="BuPu",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

plt.title('Is BCG Mandatory for All Children?')

sns.set(font_scale=1)
ax = df['vaccination_timing'].value_counts().plot.barh(figsize=(16, 8))

ax.set_title('Vaccination Timing', size=18)

ax.set_ylabel('vaccination_timing', size=10)

ax.set_xlabel('count', size=10)
fig = px.bar(df[['vaccination_timing','country_name']].sort_values('country_name', ascending=False), 

                        y = "country_name", x= "vaccination_timing", color='country_name', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='forestgreen', size=14))

fig.update_layout(title_text="BCG Vaccination Timing by Country")



fig.show()
df3 = pd.read_csv('../input/hackathon/BCG_world_atlas_data-2020.csv')

df3.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x="Definition of High-risk groups (if applicable) which receive BCG?",data=df3,palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title('BCG Vaccinated High-risk Groups')

# changing the font size

sns.set(font_scale=1)
ax = df3['Definition of High-risk groups (if applicable) which receive BCG?'].value_counts().plot.barh(figsize=(80, 80))

ax.set_title('BCG Vaccinated High-Risk Groups', FontSize = 60)

ax.set_ylabel('Definition of High-risk groups (if applicable) which receive BCG?', Fontsize = 80)

ax.set_xlabel('count', FontSize = 80)

ax.legend(fontsize=60)

#plt.xticks(rotation=45, fontsize = 50)

plt.yticks(fontsize = 80)
df1 = pd.read_csv('../input/hackathon/task_2-Tuberculosis_infection_estimates_for_2018.csv')

df1.head()
ax = df1.groupby('g_whoregion')['e_prevtx_kids_pct_hi'].mean().plot(kind='barh', figsize=(12,8),

                                                           title='Mean Estimated % of children received TB preventive therapy High Bound ')

plt.xlabel('Mean Estimated % of children received TB preventive therapy')

plt.ylabel('Who region')

plt.show()
ax = df1.groupby('g_whoregion')['e_prevtx_kids_pct_lo'].min().sort_values(ascending=True).plot(kind='barh', figsize=(12,8), color='r',

                                                                                  title='Min.Estimated % of children received TB Preventive Therapy Low Bound')

plt.xlabel('Min.Estimated % of children received TB preventive therapy')

plt.ylabel('Who region')

plt.show()
ax = df1.groupby('g_whoregion')['e_prevtx_eligible_lo', 'e_prevtx_eligible_hi'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Children under 5,household contacts of TB cases, elegible for TB Preventive Therapy')

plt.xlabel('g_whoregion')

plt.ylabel('Log Scale Children under 5, household contacts of TB cases, Eligible for TB Preventive Therapy')



plt.show()
ax = df1.groupby('g_whoregion')['e_prevtx_eligible_hi', 'e_prevtx_eligible_lo'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Children under 5,who are household contacts of TB cases eligible for TB Preventive Therapy ', logx=True, linewidth=3)

plt.xlabel('Log Scale Children under 5 who are household contacts of TB cases eligible for TB Preventive Treatment ')

plt.ylabel('Who Region')

plt.show()
ax = df['were_revaccinations_recommended'].value_counts().plot.barh(figsize=(16, 8), color='g')

ax.set_title('Were revaccinations recommended?', size=18)

ax.set_ylabel('were_revaccinations_recommended', size=10)

ax.set_xlabel('count', size=10)
ax = df['timing_of_revaccination'].value_counts().plot.barh(figsize=(16, 8), color='g')

ax.set_title('BCG Time interval of revaccination', size=18)

ax.set_ylabel('Timing of revaccination', size=10)

ax.set_xlabel('count', size=10)
fig = px.bar(df[['timing_of_revaccination','country_name']].sort_values('country_name', ascending=False), 

                        y = "country_name", x= "timing_of_revaccination", color='country_name', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="BCG Time interval of revaccination by country")



fig.show()
fig = px.bar(df, 

             x='timing_of_revaccination', y='country_name', color_discrete_sequence=['#27F1E7'],

             title='BCG Revaccination Time Interval', text='bcg_strain_original')

fig.show()
df2 = pd.read_csv('../input/hackathon/task_2-Gemany_per_state_stats_20June2020.csv', encoding='utf8')

df2.head()
plt.style.use('dark_background')

sns.countplot(x="East/West",data=df2,palette="OrRd",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

plt.title('East/West German')

sns.set(font_scale=1)
fig = px.bar(df2, 

             x='Deaths', y='East/West', color_discrete_sequence=['#27F1E7'],

             title='Covid19 Deaths in Germany', text='Population Density',  template='plotly_dark')

fig.show()
fig = px.bar(df2,

             y='East/West',

             x='Deaths',

             orientation='h',

             color='State in Germany (German)',

             title='Covid19 Cases East & West Germany',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.RdGy,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.line(df2, x="East/West", y="Deaths", color_discrete_sequence=['darkorange'], 

              title="Deaths by Covid19 in Germany", template='plotly_dark' )

fig.show()
fig = px.scatter_matrix(df2)

fig.show()