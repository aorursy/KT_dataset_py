# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/../input/suicides-in-india/Suicides in India 2001-2012.csv")

india_pop = pd.read_csv("../input/india-population/India_population.csv")

india_pop = india_pop.set_index('Year')

india_pop.head()



all_india_data = df[(df['State'] == 'Total (All India)') & (df['Type_code'] == 'Education_Status')]

yearly_suicides = all_india_data.groupby('Year')['Total'].agg({'Suicides':'sum'})

yearly_suicides

year_final = pd.merge(yearly_suicides, india_pop, right_index = True, left_index = True)



year_final['suicide_rate_per_100000'] = (year_final['Suicides']/year_final['Avg_Population'])*100000

year_final
get_ipython().magic('matplotlib inline')

import matplotlib as mpl

mpl.get_backend()

import matplotlib.pyplot as plt



#fig = plt.figure(figsize=(10,5), dpi=200)



fig, ax1 = plt.subplots(figsize=(10, 5), dpi=100)



ax1.plot(year_final.index, year_final['suicide_rate_per_100000'], color='tab:orange', 

         marker = 'o', linewidth=2, label = "Suicide Rate")

ax1.set_ylabel("Suicides per 1 million", fontsize=10, color="tab:orange")

ax1.set_ylim([0, 12])

for label in ax1.get_yticklabels():

    label.set_color("tab:orange")

    

ax2 = ax1.twinx()

ax2.bar(year_final.index, year_final['Suicides'], color='lightgrey', linewidth=1, label = 'Total Suicides')

ax2.set_ylabel("No. of Suicides", fontsize=10, color="grey")

ax2.set_ylim([0, 140000])

for label in ax2.get_yticklabels():

    label.set_color("grey")

    

ax1.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2 

ax1.patch.set_visible(False) # hide the 'canvas'

plt.title('Suicide Rate in India from 2001-2012', fontsize=12, fontweight = 'bold')



plt.xticks(year_final.index)
gender_suicides = all_india_data.groupby('Gender')['Total'].agg({'Suicides':'sum'})

gender_suicides

age_group_data = df[df['Type_code'] == 'Means_adopted']

age_suicides = age_group_data.groupby(['Age_group','Gender'])['Total'].agg({'Suicides':'sum'})

age_suicides

causes_data = df[df['Type_code'] == 'Causes']

cause_suicides = causes_data.groupby(['Type','Year'])['Total'].agg({'Suicides':'sum'})

cause_suicides.reset_index(inplace=True)



cause_suicides1 = cause_suicides.pivot(index = 'Type', columns='Year', values = 'Suicides')



cause_suicides1['total'] = cause_suicides1.sum(axis=1)

cause_suicides1['min'] = cause_suicides1.min(axis=1)

cause_suicides1['max'] = cause_suicides1.max(axis=1)

cause_suicides1['sd'] = cause_suicides1.std(axis=1)

cause_suicides1 = cause_suicides1.sort(['total'], ascending=True)

cause_suicides1.head(2)
y_pos = np.arange(len(cause_suicides1.index))



fig, ax1 = plt.subplots(figsize=(8, 8), dpi=100)

plt.barh(y_pos, cause_suicides1['total'], 0.8, alpha=0.8, color='b', xerr=cause_suicides1['sd'])

plt.yticks(y_pos, cause_suicides1.index)



ax1.xaxis.tick_top()



plt.suptitle('Causes for Snuicides in India from 2001-2012', fontsize=12, fontweight = 'bold')

plt.title('Error bar represents Std. Deviation across 12 years', fontsize=10, style='italic', color = 'green',y=1.06)

means_data = df[df['Type_code'] == 'Means_adopted']

means_suicides = means_data.groupby(['Type'])['Total'].agg({'Suicides':'sum'})

means_suicides
prof_data = df[df['Type_code'] == 'Professional_Profile']

prof_suicides = prof_data.groupby(['Type'])['Total'].agg({'Suicides':'sum'})

prof_suicides
social_data = df[df['Type_code'] == 'Social_Status']



no_states = ['Total (All India)', 'Total (States)', 'Total (Uts)']



social_data = social_data[~social_data['State'].isin(no_states)]



social_suicides = social_data.groupby(['Type'])['Total'].agg({'Suicides':'sum'})

social_suicides