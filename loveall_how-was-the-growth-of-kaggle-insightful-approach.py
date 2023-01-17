import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):

    temp = cols

    cm = sns.light_palette("green", as_cmap=True)

    return (round(pd.crosstab(df[temp[0]], df[temp[1]], 

                       normalize=normalize, values=values, aggfunc=aggfunc) * 100,2)).style.background_gradient(cmap = cm)



def date_time_col(df, col, form='%m/%d/%Y'):

    df[col] = pd.to_datetime(df[col], format=form)

    return df[col]
kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')

kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv')

kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv')

kernel_votes = pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv')

kernel_langs = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')

follow_users = pd.read_csv('/kaggle/input/meta-kaggle/UserFollowers.csv')

users_achi = pd.read_csv("/kaggle/input/meta-kaggle/UserAchievements.csv")

users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')

tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')



comp = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')
kernel_tags.rename(columns={'Id':'KernelId'}, inplace=True)

kernels.rename(columns={'Id':'KernelId'}, inplace=True)

kernel_versions.rename(columns={'Id':'KVersId', 

                                'TotalVotes':'TotalVotes_version'}, inplace=True)

kernel_votes.rename(columns={'Id':'KernVotId'}, inplace=True)

kernel_langs.rename(columns={'Id':'KernelLanguageId'}, inplace=True)

follow_users.rename(columns={'Id':'UserId'}, inplace=True)

users_achi.rename(columns={'Id':'AchId'}, inplace=True)

users.rename(columns={'Id':'UserId'}, inplace=True)

users_achi.rename(columns={'Id':'AchId'}, inplace=True)

#dataset.rename(columns={'Id':'dsetId'}, inplace=True)

kernel_versions.rename(columns={'TotalVotes':'TotalVotes_version'}, inplace=True)
resumetable(users)
resumetable(kernels)
resumetable(users_achi)
def date_time_col(df, col, form='%m/%d/%Y'):

    df[col] = pd.to_datetime(df[col], format=form)

    return df[col]



users['RegisterDate'] = date_time_col(users, 'RegisterDate')
# Calling the function to transform the date column in datetime pandas object



#seting some static color options

color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 

            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 

            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']





dates_temp = users.groupby(users['RegisterDate'].dt.date)['UserId'].count().reset_index()

# renaming the columns to apropriate names



# creating the first trace with the necessary parameters

trace = go.Scatter(x=dates_temp['RegisterDate'], y=dates_temp.UserId,

                    opacity = 0.8, line = dict(color = color_op[7]), name= 'Registrations')



tmp_count = users.groupby(users['RegisterDate'])['UserId'].count().reset_index()

tmp_count['cumsum'] = tmp_count['UserId'].cumsum()

tmp_count['cumsum'] = tmp_count['cumsum'] 



# using the new dates_temp_sum we will create the second trace

trace1 = go.Scatter(x=tmp_count.RegisterDate, line = dict(color = color_op[1]), 

                    name="CumSum Registers",

                    y=tmp_count['cumsum'], opacity = 0.8, yaxis='y2')



tmp_mean = users.groupby(users['RegisterDate'].dt.date)['UserId'].count().reset_index()

tmp_mean['mean'] = round(tmp_mean['UserId'].rolling(90).mean())



# using the new dates_temp_sum we will create the second trace

trace2 = go.Scatter(x=tmp_mean.RegisterDate, line = dict(color = color_op[3]), 

                    name="MovAverage 90 days",

                    y=tmp_mean['mean'], opacity = 0.8)





layout = dict(

    title= "Kaggle Users Registration by Dates",

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=3, label='3m', step='month', stepmode='backward'),

                dict(count=6, label='6m', step='month', stepmode='backward'),

                dict(count=12, label='12m', step='month', stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(visible = True),

        type='date' ),

    yaxis=dict(title='Total Registrations by Day'),

    yaxis2=dict(overlaying='y',

                anchor='x', side='right',

                zeroline=False, showgrid=False,

                title='Cumulative total subscribers'),

    annotations=[

        go.layout.Annotation(

            x='2017-06-11',

            y=1001168,

            xref="x",

            yref="y2",

            text="2017-06-11<br>1 Million day",

            showarrow=True,

            arrowhead=3,

            ax=-60,

            ay=-60

        ),

        go.layout.Annotation(

            x='2018-08-26',

            y=2001571,

            xref="x",

            yref="y2",

            text="2018-08-26<br>2 Million day",

            showarrow=True,

            arrowhead=3,

            ax=-80,

            ay=-40),

        go.layout.Annotation(

            x='2019-05-19',

            y=3002331,

            xref="x",

            yref="y2",

            text="2019-05-19<br>3 Million day",

            showarrow=True,

            arrowhead=3,

            ax=-80,

            ay=-30),

                

    ]

)



# creating figure with the both traces and layout

fig = dict(data= [trace, trace1, trace2],

           layout=layout)



#rendering the graphs

iplot(fig) #it's an equivalent to plt.show()
users['month_reg'] = users['RegisterDate'].dt.month

users['dow_reg'] = users['RegisterDate'].dt.dayofweek

users['day_reg'] = users['RegisterDate'].dt.day
#snsusers['dow_reg'].value_counts()

total = len(users)

plt.figure(figsize=(15,19))



plt.subplot(311)

g = sns.countplot(x="day_reg", data=users, color='coral')

g.set_title("User Registers by DAY OF MONTH", fontsize=20)

g.set_ylabel("Count",fontsize= 17)

g.set_xlabel("Day of Registrations", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)



plt.subplot(312)

g2 = sns.countplot(x="dow_reg", data=users, color='coral')

g2.set_title("User Registers by DAY OF WEEK", fontsize=20)

g2.set_ylabel("Count",fontsize= 17)

g2.set_xlabel("Day of the Week of Registrations", fontsize=17)

sizes=[]

for p in g2.patches:

    height = p.get_height()

    sizes.append(height)

    g2.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g2.set_ylim(0, max(sizes) * 1.15)



plt.subplot(313)

g1 = sns.countplot(x="month_reg", data=users, color='coral')

g1.set_title("User Registers by MONTHS", fontsize=20)

g1.set_ylabel("Performance Tier Cats",fontsize= 17)

g1.set_xlabel("", fontsize=17)

sizes=[]

for p in g1.patches:

    height = p.get_height()

    sizes.append(height)

    g1.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g1.set_ylim(0, max(sizes) * 1.15)



plt.subplots_adjust(hspace = 0.3)



plt.show()
cross_heatmap(users, 

              ['dow_reg', 'month_reg'],

              normalize='columns')
tier_dict = {0:'Novice', 1:'Contributor', 2:'Expert',

             3:'Master', 4:'GranMaster', 5:'KaggleTeam'}



users['PerformanceTier'] = users['PerformanceTier'].map(tier_dict)



plt.figure(figsize=(15,6))



g = sns.countplot(x="PerformanceTier", data=users, color='coral',

                  order=['KaggleTeam', 'Novice', 'Contributor', 

                         'Expert', 'Master', 'GranMaster'])

g.set_title("Performance Tier Distribution", fontsize=20)

g.set_ylabel("Performance Tier Cats",fontsize= 17)

g.set_xlabel("", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.4f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15)



plt.show()
users_comb = users_achi.merge(users, on='UserId')

users_comb['PerformanceTier'] = users_comb['PerformanceTier'].map(tier_dict)

users_comb['Tier'] = users_comb['Tier'].map(tier_dict)
users_comb['RegisterDate'] = date_time_col(users_comb, 'RegisterDate')

users_comb['TierAchievementDate'] = date_time_col(users_comb, 'TierAchievementDate')



users_comb['diffReg'] = (users_comb['TierAchievementDate'].dt.date - users_comb['RegisterDate'].dt.date).dt.days



table_user_tier = users_comb[(users_comb['Tier'] != 'Novice') & 

                             (users_comb['RegisterDate'] >= '2011-03-01')].groupby(['Tier', 

                                                                                    'AchievementType'])['diffReg'].mean().reset_index()

tier_count = users_comb[(users_comb['Tier'] != 'Novice') &

                        (users_comb['RegisterDate'] >= '2011-03-01')].groupby(['Tier',

                                                                               'AchievementType'])['UserId'].count().reset_index()





plt.figure(figsize=(15,11))



plt.subplot(211)

g0 = sns.barplot(x='Tier', y='UserId', 

                hue='AchievementType', data=tier_count,

                order = ['Expert', 'Master', 'GranMaster'])

g0.set_title("Distribution of each Category", fontsize=20)

g0.set_xlabel("Kaggle Category's", fontsize=17)

g0.set_ylabel("Count total", fontsize=17)

sizes=[]

for p in g0.patches:

    height = p.get_height()

    sizes.append(height)

    g0.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.0f}'.format(height),

            ha="center", fontsize=14) 

g0.set_ylim(0, max(sizes) * 1.15)



plt.subplot(212)

g = sns.barplot(x='Tier', y='diffReg', 

                hue='AchievementType', data=table_user_tier,

                order = ['Expert', 'Master', 'GranMaster'])

g.set_title("Time to take different Tiers (IN DAYS)", fontsize=20)

g.set_xlabel("Kaggle Category's", fontsize=17)

g.set_ylabel("Mean of Days", fontsize=17)



sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.0f}'.format(height),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15)



plt.subplots_adjust(hspace = 0.3)



plt.show()

users_comb = pd.merge(kernels, users, left_on='AuthorUserId', right_on='UserId', how='left')
users_comb['CreationDate'] = date_time_col(users_comb, 'CreationDate',

                                           form='%m/%d/%Y %H:%M:%S')



users_comb['MadePublicDate'] = date_time_col(users_comb, 'MadePublicDate',

                                             form='%m/%d/%Y')

users_comb['RegisterDate'] = users_comb['RegisterDate'].dt.date

users_comb['CreationDate'] = users_comb['CreationDate'].dt.date

users_comb['MadePublicDate'] = users_comb['MadePublicDate'].dt.date



# To get votes by date

kernel_versions['CreationDate'] = date_time_col(kernel_versions, 'CreationDate',

                                                  form='%m/%d/%Y %H:%M:%S')



kernel_versions['CreationDate'] = kernel_versions['CreationDate'].dt.date
# Calling the function to transform the date column in datetime pandas object



#seting some static color options

color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 

            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 

            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']





tmp_kern = users_comb.groupby('CreationDate')['KernelId'].nunique().reset_index()

# renaming the columns to apropriate names



# creating the first trace with the necessary parameters

trace = go.Scatter(x=tmp_kern['CreationDate'], y=tmp_kern.KernelId,

                    opacity = 0.8, line = dict(color = color_op[7]), 

                   name= 'Daily Creations')





tmp_kern['cumsum'] = tmp_kern['KernelId'].cumsum()



# using the new dates_temp_sum we will create the second trace

trace1 = go.Scatter(x=tmp_kern.CreationDate, line = dict(color = color_op[1]), 

                    name="Total Sum Kernels",

                    y=tmp_kern['cumsum'], opacity = 0.8, yaxis='y2')



tmp_kern['mean'] = round(tmp_kern['KernelId'].rolling(30).mean())



# using the new dates_temp_sum we will create the second trace

trace2 = go.Scatter(x=tmp_kern.CreationDate, line = dict(color = color_op[3]), 

                    name="Mov.Average 30 Days", 

                    y=tmp_kern['mean'], opacity = 0.8)





##tmp_votes = kernel_versions.groupby('CreationDate')['TotalVotes_version'].sum().reset_index()



##trace3 = go.Scatter(x=tmp_votes.CreationDate, line = dict(color = color_op[6]), 

##                    name="Mov.Average 3fds0 Days",

##                    y=tmp_votes['TotalVotes_version'], opacity = 0.8)





layout = dict(

    title= "Kernel Informations by Date",

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=3, label='3m', step='month', stepmode='backward'),

                dict(count=6, label='6m', step='month', stepmode='backward'),

                dict(count=12, label='12m', step='month', stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(visible = True),

        type='date' ),

    yaxis=dict(title='Kernel Creation by Day'),

    yaxis2=dict(overlaying='y',

                anchor='x', side='right',

                zeroline=False, showgrid=False,

                title='Cumulative Total Kernels'),



    annotations=[

            go.layout.Annotation(

                x='2017-06-11',

                y=43,

                xref="x",

                yref="y",

                text="NOTE<br>It's 1 million day. <br>what happened?",

                showarrow=True,

                arrowhead=3,

                ax=51,

                ay=-50

            )



        ]

)



# creating figure with the both traces and layout

fig = dict(data= [trace, trace1, trace2, #trace3

                 ],

           layout=layout)



#rendering the graphs

iplot(fig) 
comp['RewardType'].fillna('None', inplace=True)
total = len(comp)



print(f'Total of registered competitions on Kaggle: {total}')



plt.figure(figsize=(14,5))



g = sns.countplot(x='RewardType', data=comp, color='coral')

g.set_title("Competition Reward Types", fontsize=22)

g.set_ylabel("Count", fontsize=17)

g.set_xlabel("Reward Type Name", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

    

g.set_ylim(0, max(sizes) * 1.15)





plt.show()
comp.loc[comp["RewardType"].isin(['Knowledge', 'Swag',

                                  'Jobs', 'Kudos']), 'MoneyComp'] = 0

comp['MoneyComp'].fillna(1, inplace=True)



comp['EnabledDate'] = date_time_col(comp, 'EnabledDate',

                                           form='%m/%d/%Y %H:%M:%S')

comp['EnabledDate'] = comp['EnabledDate'].dt.date
tmp_comp = comp.groupby('EnabledDate')['Id'].count().reset_index()

money_comp = comp.loc[comp["MoneyComp"] == 1].groupby('EnabledDate')['Id'].count().reset_index()

knowledge_comp = comp.loc[comp["MoneyComp"] == 0].groupby('EnabledDate')['Id'].count().reset_index()
# Calling the function to transform the date column in datetime pandas object



#seting some static color options

color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 

            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 

            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']





#tmp_kern = comp.groupby('CreationDate')['KernelId'].nunique().reset_index()

# renaming the columns to apropriate names



# creating the first trace with the necessary parameters

trace = go.Scatter(x=money_comp['EnabledDate'], y=money_comp.Id,

                   opacity = 0.8, line = dict(color = color_op[7]), 

                   name= 'Money Comps')





tmp_comp['cumsum'] = tmp_comp['Id'].cumsum()



# using the new dates_temp_sum we will create the second trace

trace1 = go.Scatter(x=tmp_comp.EnabledDate, line = dict(color = color_op[1]), 

                    name="Total cumulative",

                    y=tmp_comp['cumsum'], opacity = 0.8, yaxis='y2')



# using the new dates_temp_sum we will create the second trace

trace2 = go.Scatter(x=knowledge_comp.EnabledDate, line = dict(color = color_op[3]), 

                    name="Knowledge_comps", 

                    y=knowledge_comp['Id'], opacity = 0.8)





##tmp_votes = kernel_versions.groupby('CreationDate')['TotalVotes_version'].sum().reset_index()



##trace3 = go.Scatter(x=tmp_votes.CreationDate, line = dict(color = color_op[6]), 

##                    name="Mov.Average 3fds0 Days",

##                    y=tmp_votes['TotalVotes_version'], opacity = 0.8)





layout = dict(

    title= "Competitions Creation Distribution by Dates",

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=3, label='3m', step='month', stepmode='backward'),

                dict(count=6, label='6m', step='month', stepmode='backward'),

                dict(count=12, label='12m', step='month', stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(visible = True),

        type='date' ),

    yaxis=dict(title='Total by the days'),

    yaxis2=dict(overlaying='y',

                anchor='x', side='right',

                zeroline=False, showgrid=False,

                title='Cumulative Total Comps')

)



# creating figure with the both traces and layout

fig = dict(data= [trace, trace1, trace2, #trace3

                 ],

           layout=layout)



#rendering the graphs

iplot(fig) 
user_votes = pd.merge(kernel_votes, users[['UserName','DisplayName', 

                                           'RegisterDate','UserId',

                                           'PerformanceTier' ]],

                      left_on='UserId', right_on='UserId', how='left')



kerneld_users_id = kernel_versions[['KVersId', 'KernelId', 'AuthorUserId']]

user_votes_2 = pd.merge(user_votes[['UserId', 'KernelVersionId',

                                    'UserName','DisplayName', 'KernVotId',

                                    'PerformanceTier']], 

                      kerneld_users_id,

                      left_on='KernelVersionId', right_on='KVersId', how='left')



user_votes_23 = pd.merge(user_votes_2, users[['UserName','DisplayName', 

                                           'RegisterDate','UserId',

                                           'PerformanceTier' ]], 

                         left_on='AuthorUserId', right_on='UserId', how='left')
tmp = user_votes_23[user_votes_23['PerformanceTier_y'] != 'KaggleTeam'].groupby(["DisplayName_y",'PerformanceTier_y']).agg({'UserName_x': 'nunique',

                                                                                                      'KernVotId': 'nunique',

                                                                                                      'KernelId': 'nunique'}).reset_index()



tmp.rename(columns={'DisplayName_y': 'User_Name',

                    'UserName_x':'Total_uniques',

                    'KernVotId': 'Total_Votes',

                    'PerformanceTier_y': 'Tier',

                    'KernelId': 'Total_Kernels'}, inplace=True)
tmp['vote_ratio'] = round(tmp['Total_Votes'] / tmp['Total_uniques'], 2)

tmp_uniq_vot = tmp[#(tmp.vote_ratio != np.inf) &

                     #(tmp.Total_uniques > 10) & 

                     (tmp.Tier.isin(['Expert', 'Master', 'GranMaster']))].sort_values('Total_uniques', ascending=False)[:30]

tmp_kernl_tot = tmp[#(tmp.vote_ratio != np.inf) &

                    # (tmp.Total_uniques > 10) & 

                     (tmp.Tier.isin(['Expert', 'Master', 'GranMaster']))].sort_values('Total_Votes', ascending=False)[:30]

tmp_kernl = tmp[#(tmp.vote_ratio != np.inf) &

                #     (tmp.Total_uniques > 10) & 

                     (tmp.Tier.isin(['Expert', 'Master', 'GranMaster']))].sort_values('Total_Kernels', ascending=False)[:30]

tmp_vote_ratio = tmp[(tmp.vote_ratio != np.inf) &

                     (tmp.Total_uniques > 10) & 

                     (tmp.Tier.isin(['Expert', 'Master', 'GranMaster']))].sort_values('vote_ratio',

                                                                                       ascending=False).head(30)
#snsusers['dow_reg'].value_counts()

plt.figure(figsize=(14,21))



plt.subplot(311)

g = sns.barplot(x="User_Name", y='Total_uniques',  

                data=tmp_uniq_vot,  color='sandybrown',

                   order=list(tmp_uniq_vot['User_Name'].values))

g.set_title("TOTAL UNIQUE VOTES \nTOP 30 USERS WITH MORE VOTE FROM UNIQUE USERS", fontsize=20)

g.set_ylabel("Count ",fontsize= 17)

g.set_xlabel("", fontsize=17)

sizes=[]

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)

g.set_xticklabels(g.get_xticklabels(),

                  rotation=90)

gt = g.twinx()

gt = sns.pointplot(x='User_Name', y='vote_ratio', data=tmp_uniq_vot, color='green',

                   order=list(tmp_uniq_vot['User_Name'].values),

                   # color='black',

                   legend=False)

gt.set_ylim(tmp_uniq_vot['vote_ratio'].min()-.3,tmp_uniq_vot['vote_ratio'].max()*1.1)

gt.set_ylabel("Mean of votes by User", fontsize=16)



plt.subplot(312)

g1 = sns.barplot(x="User_Name", y='Total_Votes',  data=tmp_kernl_tot,  color='sandybrown')

g1.set_title("TOTAL VOTES \nTOP 30 USERS WITH MORE VOTES", fontsize=20)

g1.set_ylabel("Count ",fontsize= 17)

g1.set_xlabel(" ", fontsize=17)

sizes=[]

for p in g1.patches:

    height = p.get_height()

    sizes.append(height)



g1.set_ylim(0, max(sizes) * 1.15)

g1.set_xticklabels(g1.get_xticklabels(),

                  rotation=90)

gt = g1.twinx()

gt = sns.pointplot(x='User_Name', y='vote_ratio', data=tmp_kernl_tot,

                   order=list(tmp_kernl_tot['User_Name'].values),

                   color='green', legend=False)

gt.set_ylim(tmp_kernl_tot['vote_ratio'].min()-.3,

            tmp_kernl_tot['vote_ratio'].max()*1.1)

gt.set_ylabel("Mean of votes by User", fontsize=16)





plt.subplot(313)

g2 = sns.barplot(x="User_Name", y='vote_ratio',  

                 data=tmp_vote_ratio,  color='sandybrown')

g2.set_title("UNIQUE USERS VOTE RATIO \nTOP 30 Users with the highest mean vote from unique Users", fontsize=20)

g2.set_ylabel("Count ",fontsize= 17)

g2.set_xlabel("TOP 30 Users", fontsize=17)

sizes=[]

for p in g2.patches:

    height = p.get_height()

    sizes.append(height)

    g2.text(p.get_x()+p.get_width()/2.,

            height + .05,

            f'{height}',

            ha="center", fontsize=11)



g2.set_ylim(0, max(sizes) * 1.15)

g2.set_xticklabels(g2.get_xticklabels(),

                  rotation=90)



plt.subplots_adjust(hspace = 0.8)



plt.show()
plt.figure(figsize=(14,6))

g2 = sns.barplot(x="User_Name", y='Total_Kernels',  

                 data=tmp_kernl,  color='sandybrown',

                   order=list(tmp_kernl['User_Name'].values))

g2.set_title("KERNELS  <br>TOP 30 USERS WITH MORE KERNELS", fontsize=20)

g2.set_ylabel("Count ",fontsize= 17)

g2.set_xlabel("TOP 30 Users", fontsize=17)

sizes=[]

for p in g2.patches:

    height = p.get_height()

    sizes.append(height)

    g2.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.0f}'.format(height),

            ha="center", fontsize=11)

g2.set_ylim(0, max(sizes) * 1.15)

g2.set_xticklabels(g2.get_xticklabels(),

                  rotation=90)

gt = g2.twinx()

gt = sns.pointplot(x='User_Name', y='vote_ratio', data=tmp_kernl, color='green',

                   order=list(tmp_kernl['User_Name'].values),

                   # color='black',

                   legend=False)

gt.set_ylim(tmp_kernl['vote_ratio'].min()-.3,tmp_kernl['vote_ratio'].max()*1.1)

gt.set_ylabel("Mean of votes by User", fontsize=16)



plt.show()