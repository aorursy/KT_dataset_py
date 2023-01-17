import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

df_pop = pd.read_csv('/kaggle/input/world-population-data-2018/world_population_2018.csv')



df = df.drop(0)

df.head()

#df['Q3'] = df['Q3'].replace(['United Kingdom of Great Britain and Northern Ireland'], 'UK & IR')

#df['Q3'] = df['Q3'].replace(['Hong Kong (S.A.R.)'], 'China')

df['Q3'] = df['Q3'].replace({'United Kingdom of Great Britain and Northern Ireland': 'UK & IR', 

                             'Iran, Islamic Republic of...': 'Iran', 'Viet Nam': 'Vietnam', 

                             'United States of America': 'USA'})





df_pop['Country Name'] = df_pop['Country Name'].replace({'Hong Kong SAR, China': 'Hong Kong (S.A.R.)', 

                                         'United Kingdom': 'UK & IR', 'Korea, Rep.': 'Republic of Korea',

                                        'Korea, Dem. People’s Rep.': 'North Korea',

                                         'Iran, Islamic Rep.': 'Iran', 'United States': 'USA', 

                                        'Egypt, Arab Rep.': 'Egypt', 'Russian Federation': 'Russia'})
colors = ['silver',] * 59

colors[-2:] = ['crimson' for i in colors[-2:]]

    

df['Q3'].value_counts(normalize = True).sort_values().plot(kind='barh', figsize=(15,15), color = colors, rot=0)



plt.xlabel("Percent", labelpad=14, fontsize=20)

plt.ylabel("Countries", labelpad=14, fontsize=20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.title("Percent of Countries", y=1.02, fontsize=25)
df_country = df[['Q3', 'Q1', 'Q2', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q10', 'Q11', 'Q15' ]]

df1 = df_country.groupby(['Q3'])['Q3'].count().to_frame('count').reset_index()



#merge with population



df_merge = pd.merge(df1, df_pop, left_on = 'Q3', right_on = 'Country Name')



df_merge= df_merge.sort_values('pop_2018', ascending = False)



#plt.figure(figsize=(20,15))

#sns.set_color_codes("muted")

#sns.barplot(x="pop_2018", y="Country Name", data=df_merge)
import plotly

import plotly.express as px

plotly.offline.init_notebook_mode(connected=True)

fig = px.bar(df_merge.sort_values('pop_2018',ascending = False), y="pop_2018", x="Country Name", hover_data=["pop_2018"])

fig.update_layout(title="2018 Population of Countries",

                  xaxis_title="Country",

                  yaxis_title="Population2018", xaxis_tickangle=-45)

fig.show()
import plotly.express as px

#data = px.data.df_merge()

fig = px.scatter(df_merge, x="pop_2018", y="count", color="count",

                  size='pop_2018', hover_data=['Country Name'])

fig.update_layout(title="Country Population Vs. Kaggle survey population",

                  xaxis_title="Country Population",

                  yaxis_title="Survey population")

fig.show()
df_merge['per_1M'] = round((df_merge['count']/df_merge['pop_2018'])*1000000, 0)

df_merge= df_merge.sort_values('per_1M', ascending = False).drop(columns = ['Q3'])

#df_merge
colors = ['silver',] * 59

colors[14] = 'crimson'

colors[31] = 'crimson'



plt.figure(figsize=(15,15))

ax = sns.barplot(y="Country Name", x="per_1M", data=df_merge, palette = colors)

plt.xlabel("Number of respondents per million", labelpad=14, fontsize=20)

plt.ylabel("Countries", labelpad=14, fontsize=20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.title("Number of respondents per million", y=1.02, fontsize=25)
top_15 = df[df['Q3'].isin(['Singapore','Ireland', 'Canada', 'Israel', 

                            'Switzerland', 'Portugal', 'Australia', 

                            'New Zealand', 'Greece', 'Norway', 'Denmark', 

                            'USA', 'Netherlands', 'Spain', 'Hong Kong (S.A.R)'])]
least_15 = df[df['Q3'].isin(['China','Indonesia', 'Philippines', 'Bangladesh', 

                            'Thailand', 'Pakistan', 'Egypt', 

                            'Iran', 'Vietnam', 'Republic of Korea', 'Algeria', 

                            'Saudi Arabia', 'Kenya', 'Nigeria', 'South Africa'])]
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)





top_15['Q2'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(15, 10))

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14, fontsize=20)

ax1.tick_params(labelsize=15)

ax1.set_title("Gender in most engaged", y=1.02,fontsize=20)



least_15['Q2'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(15, 10))

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14, fontsize=20)

ax2.tick_params(labelsize=15)

ax2.set_title("Gender in least engaged", y=1.02,fontsize=20)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)



colors = ['silver',] * 11

#colors[-3:] = ['crimson' for i in colors[-3:]]

colors[3] = 'green'



top_15['Q1'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(15, 10), color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("Age", labelpad=14, fontsize=20)

ax1.tick_params(labelsize=15)

ax1.set_title("Age in most_engaged", y=1.02,fontsize=20)





colors = ['silver',] * 11

colors[8] = 'green'

least_15['Q1'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(15, 10), color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("Age", labelpad=14, fontsize=20)

ax2.tick_params(labelsize=15)

ax2.set_title("Age in least_engaged", y=1.02,fontsize=20)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)



colors = ['silver',] * 7

colors[4] = 'green'

top_15['Q4'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)



ax1.set_xlabel("Percent", labelpad=14, fontsize=35)

ax1.set_ylabel("Education", labelpad=14, fontsize=35)

ax1.tick_params(labelsize=40)

ax1.set_title("Education in top15", y=1.02,fontsize=35)





colors = ['silver',] * 7

colors[4] = 'green'

least_15['Q4'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)



ax2.set_xlabel("Percent", labelpad=14, fontsize=35)

ax2.set_ylabel("Education", labelpad=14, fontsize=35)

ax2.tick_params(labelsize=40)

ax2.set_title("Education in least15", y=1.02,fontsize=35)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)



colors = ['silver',] * 12

colors[10] = 'yellow'

top_15['Q5'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 29), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14, fontsize=20)

ax1.tick_params(labelsize=20)

ax1.set_title("Job title in most_engaged", y=1.02,fontsize=25)





colors = ['silver',] * 12

colors[11] = 'yellow'

least_15['Q5'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14, fontsize=20)

ax2.tick_params(labelsize=20)

ax2.set_title("Job title in least_engaged", y=1.02,fontsize=25)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)





colors = ['silver',] * 5

colors[4] = 'orange'

top_15['Q6'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14, fontsize=20)

ax1.tick_params(labelsize=20)

ax1.set_title("Company size in most_engaged", y=1.02,fontsize=25)





colors = ['silver',] * 5

colors[0] = 'orange'

least_15['Q6'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14, fontsize=20)

ax2.tick_params(labelsize=20)

ax2.set_title("Company size in least_engaged", y=1.02,fontsize=25)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)





colors = ['silver',] * 6

colors[-2:] = ['plum' for i in colors[-2:]]

top_15['Q8'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(30, 30), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=40)

ax1.set_ylabel("", labelpad=14, fontsize=40)

ax1.tick_params(labelsize=40)

ax1.set_title("ML methods in most_engaged", y=1.02,fontsize=40)





colors = ['silver',] * 6

colors[-2:] = ['plum' for i in colors[-2:]]

least_15['Q8'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=40)

ax2.set_ylabel("", labelpad=14, fontsize=40)

ax2.tick_params(labelsize=40)

ax2.set_title("ML methods used in least_engaged", y=1.02,fontsize=40)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)



colors = ['silver',] * 25

colors[-3:] = ['violet' for i in colors[-3:]]

top_15['Q10'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14)

ax1.tick_params(labelsize=20)

ax1.set_title("Salary in most_engaged", y=1.02,fontsize=20)





colors = ['silver',] * 25

colors[-2:] = ['violet' for i in colors[-2:]]

least_15['Q10'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14)

ax2.tick_params(labelsize=20)

ax2.set_title("Salary in least_engaged", y=1.02,fontsize=20)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)



colors = ['silver',] * 6

colors[-2] = 'limegreen'

top_15['Q11'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14)

ax1.tick_params(labelsize=20)

ax1.set_title("ML spending in most_engaged", y=1.02,fontsize=20)



colors = ['silver',] * 6

colors[-2] = 'limegreen'

least_15['Q11'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14)

ax2.tick_params(labelsize=20)

ax2.set_title("ML spending in least_enaged", y=1.02,fontsize=20)
fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)





colors = ['silver',] * 7

colors[4] = 'c'

colors[6] = 'c'

top_15['Q15'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14)

ax1.tick_params(labelsize=20)

ax1.set_title("coding experience in most_engaged", y=1.02,fontsize=20)







colors = ['silver',] * 7

colors[4] = 'c' 

least_15['Q15'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax2.set_xlabel("Percent", labelpad=14, fontsize=20)

ax2.set_ylabel("", labelpad=14)

ax2.tick_params(labelsize=20)

ax2.set_title("coding experience in least_engaged", y=1.02,fontsize=20)
Q18 = df[['Q3','Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4','Q18_Part_5', 'Q18_Part_6',

            'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12']]



Q18_melt = pd.melt(Q18, id_vars=['Q3'], 

        value_vars=['Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4','Q18_Part_5', 'Q18_Part_6',

            'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12'], 

        value_name='favorite programming languages do you use on a regular basis')



Q18_melt = Q18_melt.drop('variable', axis = 1)

Q18_melt.head()



Q18_melt_cat1 = Q18_melt[Q18_melt.Q3 .isin(['Singapore','Ireland', 'Canada', 'Israel', 

                            'Switzerland', 'Portugal', 'Australia', 

                            'New Zealand', 'Greece', 'Norway', 'Denmark', 

                            'USA', 'Netherlands', 'Spain', 'Hong Kong (S.A.R)'])]



Q18_melt_cat2 = Q18_melt[Q18_melt.Q3 .isin(['China','Indonesia', 'Philippines', 'Bangladesh', 

                            'Thailand', 'Pakistan', 'Egypt', 

                            'Iran', 'Vietnam', 'Republic of Korea', 'Algeria', 

                            'Saudi Arabia', 'Kenya', 'Nigeria', 'South Africa'])]





#fig, (ax1, ax2) = plt.subplots(1, 2)

fig = plt.figure()



ax1 = fig.add_subplot(221)



ax2 = fig.add_subplot(224)





colors = ['silver',] * 11

colors[8] = 'c'

Q18_melt_cat1['favorite programming languages do you use on a regular basis'].value_counts(normalize=True).sort_values().plot(ax = ax1, kind='barh', figsize=(20, 20), rot=0, color = colors)

ax1.set_xlabel("Percent", labelpad=14, fontsize=20)

ax1.set_ylabel("", labelpad=14)

ax1.tick_params(labelsize=20)

ax1.set_title("programming language in most engaged", y=1.02,fontsize=20)





colors = ['silver',] * 11

colors[3] = 'c'

Q18_melt_cat2['favorite programming languages do you use on a regular basis'].value_counts(normalize=True).sort_values().plot(ax = ax2, kind='barh', figsize=(20, 20), rot=0, color = colors)

plt.xlabel("Percent", labelpad=14, fontsize=20)

plt.ylabel("", labelpad=14)

ax2.tick_params(labelsize=20)

plt.title("programming language in least engaged", y=1.02,fontsize=20)