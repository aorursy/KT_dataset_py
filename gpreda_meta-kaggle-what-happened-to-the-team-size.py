import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

from bubbly.bubbly import bubbleplot 

from __future__ import division

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode()

IS_LOCAL = False

import os

if(IS_LOCAL):

    PATH="../input/meta-kaggle"

else:

    PATH="../input/meta-kaggle"

print(os.listdir(PATH))
competition_df = pd.read_csv(os.path.join(PATH,"Competitions.csv"))

teams_df = pd.read_csv(os.path.join(PATH,"Teams.csv"))

team_membership_df = pd.read_csv(os.path.join(PATH,"TeamMemberships.csv"))
print("Meta Kaggle competition data -  rows:",competition_df.shape[0]," columns:", competition_df.shape[1])

print("Meta Kaggle teams data -  rows:",teams_df.shape[0]," columns:", teams_df.shape[1])

print("Meta Kaggle team memberships data -  rows:",team_membership_df.shape[0]," columns:", team_membership_df.shape[1])
competition_df.head()
competition_df.describe()
competition_df["DeadlineYear"] = pd.to_datetime(competition_df['DeadlineDate']).dt.year
tmp = competition_df.groupby('HostSegmentTitle')['Id'].nunique()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()

trace = go.Bar(

    x = df['HostSegmentTitle'],y = df['Competitions'],

    name="Competitions",

    marker=dict(color="blue"),

    text=df['HostSegmentTitle']

)

data = [trace]

layout = dict(title = 'Competitions per type',

          xaxis = dict(title = 'Competitioon Type', showticklabels=True), 

          yaxis = dict(title = 'Number of competitions'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
var = ["DeadlineDate", "DeadlineYear", "CompetitionTypeId", "HostSegmentTitle", "TeamMergerDeadlineDate", "TeamModelDeadlineDate", "MaxTeamSize", "BanTeamMergers"]

competition_df[var].head(5)
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))

missing_data(competition_df[var])
competition_df.loc[competition_df['MaxTeamSize'].isnull(),'MaxTeamSize'] = -1
tmp = competition_df.groupby('DeadlineYear')['MaxTeamSize'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
dataset = df[df['MaxTeamSize']>-1]

max_team_sizes = (dataset.groupby(['MaxTeamSize'])['MaxTeamSize'].nunique()).index

data = []

for max_team_size in max_team_sizes:

    dts = dataset[dataset['MaxTeamSize']==max_team_size]

    trace = go.Bar(

        x = dts['DeadlineYear'],y = dts['Competitions'],

        name=max_team_size,

        text=('Max team size:{}'.format(max_team_size))

    )

    data.append(trace)

    

layout = dict(title = 'Number of competitions with size of max team set per year',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of competitions'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
tmp = competition_df[competition_df['MaxTeamSize']>-1]['DeadlineYear'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()

trace = go.Bar(

    x = df['index'],y = df['Competitions'],

    name='Competition',

    marker=dict(color="red")

)

data = [trace]

    

layout = dict(title = 'Total number of competitions with size of max team set per year',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of competitions'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
tmp = competition_df['DeadlineYear'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()

trace = go.Bar(

    x = df['index'], y = df['Competitions'],

    name='Competition',

    marker=dict(color="blue")

)

data = [trace]    

layout = dict(title = 'Total number of competitions per year',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of competitions'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
tmp = competition_df.groupby(['DeadlineYear','MaxTeamSize'])['HostSegmentTitle'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()

df['CompLog'] = np.log(df['Competitions'] + 1)

hover_text = []

for index, row in df.iterrows():

    hover_text.append(('Year: {}<br>'+

                      'Competitions: {}<br>'+

                      'Max Team Size: {}<br>'+

                      'Competition type: {}').format(row['DeadlineYear'],

                                                row['Competitions'],

                                                row['MaxTeamSize'],

                                                row['HostSegmentTitle']

                                            ))

df['hover_text'] = hover_text

competition_type = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index

data = []

for comptype in competition_type:

    dfL = df[df['HostSegmentTitle']==comptype]

    trace = go.Scatter3d(

        x = dfL['DeadlineYear'],y = dfL['CompLog'],

        z = dfL['MaxTeamSize'],

        name=comptype,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=0.01,

            size=dfL['CompLog'] + 2,

        ),

        mode = "markers",

        text=dfL['hover_text'],

    )

    data.append(trace)

    

layout = go.Layout(title = 'Number of competitions per year and maximum team size, grouped by competition type',

         scene = dict(

                xaxis = dict(title = 'Year'),

                yaxis = dict(title = 'Competitions [log scale]'), 

                zaxis = dict(title = 'Maximum Team Size'),

                hovermode = 'closest',  

         )

                  )

fig = dict(data = data, layout = layout)

iplot(fig, filename='competitions-year-comp_type-maxteamsize')
tmp = competition_df[competition_df['HostSegmentTitle']!='InClass']

tmp = tmp.groupby(['DeadlineYear','RewardQuantity'])['HostSegmentTitle'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()

df['RewardLog'] = np.log(df['RewardQuantity'] + 1)

hover_text = []

for index, row in df.iterrows():

    hover_text.append(('Year: {}<br>'+

                      'Competitions: {}<br>'+

                      'Reward: {}<br>'+

                      'Competition type: {}').format(row['DeadlineYear'],

                                                row['Competitions'],

                                                row['RewardQuantity'],

                                                row['HostSegmentTitle']

                                            ))

df['hover_text'] = hover_text

competition_type = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index

data = []

for comptype in competition_type:

    dfL = df[df['HostSegmentTitle']==comptype]

    trace = go.Scatter3d(

        x = dfL['DeadlineYear'],y = dfL['Competitions'],

        z = dfL['RewardLog'],

        name=comptype,

        marker=dict(

            symbol='circle',

            sizemode='area',

            sizeref=0.01,

            size=dfL['RewardLog'] + 2,

        ),

        mode = "markers",

        text=dfL['hover_text'],

    )

    data.append(trace)

    

layout = go.Layout(title = 'Number of competitions per year and reward amount, grouped by competition type',

         scene = dict(

                xaxis = dict(title = 'Year'),

                yaxis = dict(title = 'Competitions'), 

                zaxis = dict(title = 'Reward Quantity [log scale]'),

                hovermode = 'closest',  

         ))

fig = dict(data = data, layout = layout)

iplot(fig, filename='competitions-year-comp_type-reward')
#exclude "inClass"

tmp = competition_df[competition_df['HostSegmentTitle']!='InClass']

tmp = tmp.groupby(['DeadlineYear','HostSegmentTitle'])['RewardQuantity'].sum()

df = pd.DataFrame(data={'Total amount': tmp.values}, index=tmp.index).reset_index()
host_segment_titles = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index

data = []

for host_segment_title in host_segment_titles:

    dts = df[df['HostSegmentTitle']==host_segment_title]

    trace = go.Bar(

        x = dts['DeadlineYear'],y = dts['Total amount'],

        name=host_segment_title,

        text=('Competition type:{}'.format(host_segment_title))

    )

    data.append(trace)



layout = dict(title = ('Total amount of reward per year, grouped by Competition type'),

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Total amount'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
teams_df.head(5)
teams_df.describe()
missing_data(teams_df)
print(f"Teams: {teams_df.shape[0]} different teams: {teams_df.Id.nunique()}")
team_membership_df.head(5)
team_membership_df.describe()
missing_data(team_membership_df)
comp_team_df = competition_df.merge(teams_df, left_on='Id', right_on='CompetitionId', how='inner')

comp_team_membership_df = comp_team_df.merge(team_membership_df, left_on='Id_y', right_on='TeamId', how='inner')
tmp = comp_team_membership_df.groupby(['DeadlineYear','TeamId'])['Id'].count()

df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

tmp = df.groupby(['DeadlineYear','Teams']).count()

df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df2.columns = ['Year', 'Team size','Teams']
def plot_heatmap_count(data_df,feature1, feature2, color, title):

    matrix = data_df.pivot(feature1, feature2, 'Teams')

    fig, (ax1) = plt.subplots(ncols=1, figsize=(16,6))

    sns.heatmap(matrix, 

        xticklabels=matrix.columns,

        yticklabels=matrix.index,ax=ax1,linewidths=.1,linecolor='darkblue',annot=True,cmap=color)

    plt.title(title, fontsize=14)

    plt.show()
tmp = comp_team_df['DeadlineYear'].value_counts()

df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

trace = go.Bar(

    x = df['index'], y = df['Teams'],

    name='Team',

    marker=dict(color="blue")

)

data = [trace]    

layout = dict(title = 'Total number of teams per year',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of teams'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='team-types')
tmp = comp_team_df.groupby('DeadlineYear')['HostSegmentTitle'].value_counts()

df = pd.DataFrame(data={'Competitions': tmp.values}, index=tmp.index).reset_index()
host_segment_titles = (df.groupby(['HostSegmentTitle'])['HostSegmentTitle'].nunique()).index

data = []

for host_segment_title in host_segment_titles:

    dts = df[df['HostSegmentTitle']==host_segment_title]

    trace = go.Bar(

        x = dts['DeadlineYear'],y = dts['Competitions'],

        name=host_segment_title,

        text=('Competition type:{}'.format(host_segment_title))

    )

    data.append(trace)



layout = dict(title = ('Number of teams per year, grouped by HostSegmentTitle (Competition type)'),

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Competitions'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')
df2.head()
plot_heatmap_count(df2,'Team size','Year', 'Greens', "Number of teams grouped by year and by team size")
no_inclass_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']!='InClass']

tmp = no_inclass_df.groupby(['DeadlineYear','TeamId'])['Id'].count()

df = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

tmp = df.groupby(['DeadlineYear','Teams']).count()

df2 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df2.columns = ['Year', 'Team size','Teams']

plot_heatmap_count(df2,'Team size','Year', 'Blues', "Number of teams grouped by year and by team size (no InClass comp.)")
tmp = comp_team_membership_df.groupby(['DeadlineYear','TeamId', 'HostSegmentTitle'])['Id'].count()

df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

tmp = df3.groupby(['DeadlineYear','HostSegmentTitle','Teams']).count()

df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df4.columns = ['Year', 'Host Segment Title','Team size','Teams']

df4['TeamsSqrt'] = np.sqrt(df4['Teams'] + 2)
figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Team size',

    bubble_column = 'Host Segment Title', time_column='Year', size_column = 'TeamsSqrt',

    x_title='Team size', y_title='Number of Teams [log scale]', 

    title='Number of Teams vs. Team size - time variation (years)', 

    colorscale='Rainbow', colorbar_title='Team size', 

    x_range=[-5,41], y_range=[-0.4,7], y_logscale=True, scale_bubble=5, height=650)

iplot(figure, config={'scrollzoom': True})
feature_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']=='Featured']
tmp = feature_df.groupby(['DeadlineYear','TeamId', 'Medal'])['Id'].count()

df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

tmp = df3.groupby(['DeadlineYear','Medal','Teams']).count()

df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df4.columns = ['Year', 'Medal','Team size','Teams']

df4['Rank'] = (df4['Medal'] - 1) / 2

df4['Size'] = 4 - df4['Medal']
#create the bins for gold, silver and bronze

bins = [-0.01, 0.49, 0.99, np.inf]

names = ['Gold', 'Silver', 'Bronze']

df4['MedalName'] = pd.cut(df4['Rank'], bins, labels=names)
figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Rank',

    bubble_column = 'MedalName', time_column='Year', size_column = 'Size', 

    x_title='Team size', y_title='Number of Teams [log scale]', 

    colorscale = [[0, "gold"], [0.5, "silver"], [1,"brown"]],

    title='Number of Winning Teams vs. Team size - time variation (years)', 

    x_range=[-5,41], y_range=[-0.4,4], y_logscale=True, scale_bubble=0.2, height=650)

iplot(figure, config={'scrollzoom': True})
df5 = df4[df4['Medal']==1.0]

plot_heatmap_count(df5,'Team size','Year', 'Greens', "Number of Gold winning teams grouped by year and by team size (Featured competitions)")
research_df = comp_team_membership_df[comp_team_membership_df['HostSegmentTitle']=='Research']
tmp = research_df.groupby(['DeadlineYear','TeamId', 'Medal'])['Id'].count()

df3 = pd.DataFrame(data={'Teams': tmp.values}, index=tmp.index).reset_index()

tmp = df3.groupby(['DeadlineYear','Medal','Teams']).count()

df4 = pd.DataFrame(data=tmp.values, index=tmp.index).reset_index()

df4.columns = ['Year', 'Medal','Team size','Teams']

df4['Rank'] = (df4['Medal'] - 1) / 2

df4['Size'] = 4 - df4['Medal']

bins = [-0.01, 0.49, 0.99, np.inf]

names = ['Gold', 'Silver', 'Bronze']

df4['MedalName'] = pd.cut(df4['Rank'], bins, labels=names)
figure = bubbleplot(dataset=df4, x_column='Team size', y_column='Teams', color_column = 'Rank',

    bubble_column = 'MedalName', time_column='Year', size_column = 'Size', 

    x_title='Team size', y_title='Number of Teams [log scale]', 

    colorscale = [[0, "gold"], [0.5, "silver"], [1,"brown"]],

    title='Number of Winning Teams vs. Team size - time variation (years) - Research competitions', 

    x_range=[-5,25], y_range=[-0.3,3], y_logscale=True, scale_bubble=0.2, height=650)

iplot(figure, config={'scrollzoom': True})
df5 = df4[df4['Medal']==1.0]

plot_heatmap_count(df5,'Team size','Year', 'Reds', "Number of Gold winning teams grouped by year and by team size (Research competitions)")
tmp = feature_df.groupby(['TeamId'])['Id'].count()

df = pd.DataFrame(data={'Team Size': tmp.values}, index=tmp.index).reset_index()

#merge back df with teams_df

df2 = df.merge(teams_df, left_on='TeamId', right_on='Id', how='inner')

var = ['Team Size', 'PublicLeaderboardRank', 'PrivateLeaderboardRank']

teams_ranks_df = df2[var]
corr = teams_ranks_df.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,

            cmap="YlGnBu",linewidths=.1,annot=True,vmin=-1, vmax=1)

plt.show()
df2["Year"] = pd.to_datetime(df2['LastSubmissionDate']).dt.year

var = ['Team Size', 'PublicLeaderboardRank', 'PrivateLeaderboardRank' ]

years = df2['Year'].unique()

years = np.sort(years[~np.isnan(years)])
f, ax = plt.subplots(5,2, figsize=(12,22))

for i, year in enumerate(years):

    teams_ranks_df = df2[df2['Year']==year]

    corr = teams_ranks_df[var].corr()

    labels = ['Size', 'Public', 'Private']

    axi = ax[i//2, i%2]

    s1 = sns.heatmap(corr,xticklabels=labels,yticklabels=labels,

                     cmap="YlGnBu",linewidths=.1,annot=True,vmin=-1, vmax=1,ax=axi)

    s1.set_title("Year: {}".format(int(year)))

plt.show()
users_df = pd.read_csv(os.path.join(PATH,"Users.csv"))
users_df.head()
users_df['RegisterYear'] = pd.to_datetime(users_df['RegisterDate'], format='%m/%d/%Y').dt.year
tmp = users_df['RegisterYear'].value_counts()

df = pd.DataFrame(data={'Users': tmp.values}, index=tmp.index).reset_index()

trace = go.Bar(

    x = df['index'],y = df['Users'],

    name='Users',

    marker=dict(color="red")

)

data = [trace]

    

layout = dict(title = 'Total number of new users per year',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of new users'),

          hovermode = 'closest',

         width = 600, height = 600

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='competition-types')