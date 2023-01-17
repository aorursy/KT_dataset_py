import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import seaborn as sns

from datetime import datetime as dt



%matplotlib inline
kickstarter = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv', parse_dates=['deadline', 'launched'], index_col='ID')

print(kickstarter.shape)

kickstarter.head(10)
kickstarter.describe(percentiles=[0.1,0.25,0.75,0.9])
kickstarter[['goal','usd_goal_real','pledged','usd pledged','usd_pledged_real','backers']].agg(['min', 'median', 'max', 'mean', 'std']).round().T
kickstarter.info()
kickstarter.nunique()
kickstarter.dtypes
column_name = 'main_category'

pd.DataFrame({'count': kickstarter[column_name].value_counts(normalize=False),

              'percent': kickstarter[column_name].value_counts(normalize=True)

             })
kickstarter['project_duration'] = kickstarter['deadline'].dt.date - kickstarter['launched'].dt.date

kickstarter['is_success'] = (kickstarter['state']=='successful')

kickstarter['duration_in_month'] = kickstarter['project_duration'].dt.days // 30

kickstarter['opening_month'] = kickstarter['launched'].dt.month

kickstarter['pledged_of_goal_usd'] = kickstarter['usd_pledged_real'].divide(kickstarter['usd_goal_real'])

kickstarter['pledged_of_goal_usd'].agg(['min', 'median', 'max', 'mean'])
kickstarter['goal']=kickstarter['goal'].astype(int)

kickstarter['usd_goal_real']=kickstarter['usd_goal_real'].astype(int)

kickstarter['usd_pledged_real']=kickstarter['usd_pledged_real'].astype(int)
#drop too small goal

tiny_goals=kickstarter[kickstarter['usd_goal_real']<=500]

print(kickstarter.shape)

kickstarter = kickstarter.drop(index=tiny_goals.index)

print(kickstarter.shape)
#drop date invalid

mask = (kickstarter['launched'].dt.year < 2000)

print(kickstarter.shape)

kickstarter = kickstarter.loc[~mask,:]

print(kickstarter.shape)
#drop unknown country

country_n0=kickstarter[kickstarter['country']=='N,0"']

print(kickstarter.shape)

kickstarter = kickstarter.drop(index=country_n0.index)

print(kickstarter.shape)
# drop no name project

no_name = kickstarter[kickstarter['name'].isna()]

print(kickstarter.shape)

kickstarter = kickstarter.drop(index=no_name.index)

print(kickstarter.shape)
kickstarter.head(10)
pie_ax=kickstarter['state'].value_counts(normalize=True).plot.pie(figsize=(9,9),explode=(0,0.07,0,0,0),

                                                              legend=True,shadow=True, autopct='%1.1f%%' ,

                                                              pctdistance=0.75, radius=1.05, wedgeprops = {'linewidth': 0.1}, 

                                                              textprops = {'fontsize': 14})

pie_ax.set_title('Projects state distribution', fontsize=16, weight='bold')

pie_ax.legend(fancybox=True, shadow=True, title='State', fontsize=11)

pie_ax.set_ylabel("")

#plt.rcParams['font.size'] = 15



plt.show()
fig = plt.figure(figsize=(14,7))

ax=kickstarter.groupby('main_category').size().sort_values(ascending=False).plot.bar(color=sns.hls_palette(15, l=0.5, s=0.5))

ax.set_ylabel('Number of projects')

ax.set_xlabel('Main Category')

ax.set_title('Projects distribution by main category', fontsize=16, weight='bold')



plt.show()
state_per_main_category = (kickstarter

                           .groupby('main_category')['state']

                           .value_counts(normalize=True)

                           .unstack().round(2)

                           .drop(columns='live')

                           .sort_values(by=['successful','failed'],axis=0, ascending=False)

                          )



ax=state_per_main_category.plot.bar(figsize=(22,8))

ax.set_title('State distribution by main category', fontsize=16, weight='bold')

ax.set_xlabel("")

ax.set_ylabel("State distribution %")

plt.show()
fig = plt.figure(figsize=(14,7))

ax = fig.gca()

ax.set_title('Success rate by main category', fontsize=16, weight='bold')

ax=kickstarter.groupby('main_category')['is_success'].mean().round(2).sort_values(ascending=False).plot.bar(color=sns.hls_palette(15, l=0.5, s=0.5))

ax.set_ylabel('Success rate')

ax.set_xlabel('Main Category')

plt.show()
success_per_category = kickstarter.groupby(['main_category', 'category'])['is_success'].mean()

main_categories_list = list(kickstarter.groupby('main_category').size().sort_values().index)



fig = plt.figure(figsize=(40,90))

gs = fig.add_gridspec(8,2)



list_of_axes = [fig.add_subplot(gs[r,c],xlim=(0,0.8)) for r in range(7) for c in [0,1] ]

list_of_axes.append(fig.add_subplot(gs[7,:],xlim=(0,0.8)))

for axes in list_of_axes:

    cat = main_categories_list.pop(0)

    axes.set_title(cat,size=25,weight='semibold')

    axes.tick_params(axis="both", labelsize=22.0)

    axes.barh(success_per_category[cat].sort_values().index,success_per_category[cat].sort_values().values,color=sns.color_palette("muted",len(success_per_category[cat])))

    

fig.subplots_adjust(hspace=0.35)

fig.tight_layout(rect=[0, 0.05, 1, 0.95])

fig.suptitle('Success rate by category in each main category',size=30,weight='bold',va='center')



plt.xlabel("Success Rate",size=20,style='oblique',weight='bold')

plt.show()
#num of projects per country



ax=kickstarter.groupby('country').size().sort_values(ascending=False).plot.bar(figsize=(16,8),color=sns.hls_palette(22, l=0.5, s=0.5))

ax.set_title('Projects in the DB by country', fontsize=16, weight='bold')

for p in ax.patches: 

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.text(x+width/2, 

            y+height+5000, 

            '{:.0f}'.format(height), 

            horizontalalignment='center', 

            verticalalignment='center')#,

            #fontsize=11

           #)
# Country Success dist - Normalized by row:

country_success_dist=kickstarter.groupby(['country','state'])['state'].count().unstack().div(

    kickstarter.groupby(['country','state'])['state'].count().unstack().sum(axis=1), axis=0).sort_values(by=['successful'], ascending=False)

country_success_dist
#success distruption by country

ax = country_success_dist.plot.bar(stacked=True, figsize=(18,9))

ax.set_title('State distruption by country', fontsize=16, weight='bold')

ax.legend(bbox_to_anchor=(1, 0.5))

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    if height<=0.01:

        continue

    x, y = p.get_xy() 

    ax.text(x+width/2, 

            y+height/2, 

            '{:.0f} %'.format(height*100), 

            horizontalalignment='center', 

            verticalalignment='center')
# Only successful

ax=country_success_dist['successful'].sort_values(ascending=False).plot.bar(figsize=(16,8), color='tab:red')

ax.set_title('Success distruption by country', fontsize=14)

plt.show()
succesful_projects=kickstarter[kickstarter['state']=='successful']

failed_projects=kickstarter[kickstarter['state']=='failed']
country_pledged_of_goal=kickstarter.groupby('country')['pledged_of_goal_usd'].agg(['mean', 'median', 'max','count']).sort_values(by='median',ascending=False)

country_pledged_of_goal
# Pledged_of_goal_usd by Country

country_order=list(country_pledged_of_goal.index.values)

plt.figure(figsize = (24,8))

ax=sns.boxenplot(data=kickstarter, x='country', y='pledged_of_goal_usd',order=country_order)

ax.set_ylim(0,5)

ax.set_ylabel("Pledged of goal",fontsize=14)

ax.set_xlabel("Country",fontsize=14)

ax.set_title('Money pledged out of goal by country', fontsize=16, weight='bold')

plt.show()
# Pledged_of_goal_usd by Country divided by Successful/Failed

fig=plt.figure(figsize=(24,12))#, constrained_layout=True)

#fig.set_size_inches(24,12)



ax1 = fig.add_subplot(2, 1, 1)

ax1=sns.boxenplot(data=succesful_projects, x='country', y='pledged_of_goal_usd',order=country_order)

ax1.set_title('Succesful Projects')

ax1.set_ylabel('Pledged of goal')

ax1.set_ylim(1,10)

fig.suptitle('Money pledged out of goal by country, divided to successful and failed project', fontsize=16, weight='bold')



ax2 = fig.add_subplot(2, 1, 2)

ax2=sns.boxenplot(data=failed_projects, x='country', y='pledged_of_goal_usd',order=country_order)

ax2.set_title('Failed Projects')

ax2.set_ylabel('Pledged of goal')

ax2.set_ylim(0,1)



#fig.tight_layout()

plt.show()
# summary of succesful_projects

succesful_projects['pledged_of_goal_usd'].agg(['min', 'median', 'max', 'mean','count'])
# summary of failed_projects

failed_projects['pledged_of_goal_usd'].agg(['min', 'median', 'max', 'mean','count'])
failed_but_pledged=failed_projects[failed_projects['pledged_of_goal_usd']>1]

print(failed_but_pledged.shape)

failed_but_pledged.head(5)
top_currencies=kickstarter.groupby(['currency','state'])['state'].count().unstack().sort_values(by=['successful'],ascending=False).head(5)

top_currencies
#normalized by currency

ax=top_currencies.div(top_currencies.sum(axis=1), axis=0).sort_values(by=['successful'], ascending=False).plot.bar(stacked=True, figsize=(18,9))

ax.set_title('State distribution by currency', fontsize=16, weight='bold')

ax.legend(bbox_to_anchor=(1, 0.5))

#add percentage values on stacked bars

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    if height<=0.01:

        continue

    x, y = p.get_xy() 

    ax.text(x+width/2, 

            y+height/2, 

            '{:.0f} %'.format(height*100), 

            horizontalalignment='center', 

            verticalalignment='center',

            fontsize=14)
non_us=kickstarter[kickstarter['country'] != 'US']

print(non_us.shape)

non_us.head()
non_us.groupby(['currency','state'])['state'].count().unstack().sort_values(by='successful', ascending=False)
kickstarter['goal_category']=pd.cut(kickstarter['usd_goal_real'],[0,2500,10000,25000,100000001], labels=['low', 'medium','high', 'very_high'])

succesful_projects=kickstarter[kickstarter['state']=='successful']

failed_projects=kickstarter[kickstarter['state']=='failed']
kickstarter['usd_goal_real'].describe().round()
kickstarter['goal_category'].value_counts()
goal_success_dist=kickstarter.groupby(['goal_category','state'])['state'].count().unstack().div(

    kickstarter.groupby(['goal_category','state'])['state'].count().unstack().sum(axis=1), axis=0).sort_values(by=['successful'], ascending=False)

goal_success_dist
ax = goal_success_dist.plot.bar(stacked=True, figsize=(18,9))

ax.set_title('State distribution by goal groups', fontsize=16, weight='bold')

ax.set_xlabel('Goal category')

#add percentage values on stacked bars

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    if height<=0.02:

        continue

    x, y = p.get_xy() 

    ax.text(x+width/2, 

            y+height/2, 

            '{:.0f} %'.format(height*100), 

            horizontalalignment='center', 

            verticalalignment='center',

            fontsize=14)
fig, axes = plt.subplots(1, 2)

fig.set_size_inches(30,8)



#succesful

sns.boxplot(data=succesful_projects, x='goal_category', y='pledged_of_goal_usd', ax=axes[0])

axes[0].set_title('Succesful Projects', fontsize=14)

axes[0].set_ylabel('Pledged of goal')

axes[0].set_ylim(1,5)

fig.suptitle('Money pledged out of goal by goal groups, divided to successful and failed project', fontsize=18, weight='bold')



# failed

sns.boxplot(data=failed_projects, x='goal_category', y='pledged_of_goal_usd', ax=axes[1])

axes[1].set_title('Failed Projects', fontsize=14)

axes[1].set_ylabel('Pledged of goal')

axes[1].set_ylim(0,1)



#fig.tight_layout()

plt.show()
plt.figure(figsize=(10,8))



success = pd.DataFrame({'0': kickstarter[kickstarter['duration_in_month']==0]['state'].value_counts(normalize=True)})

success.index.name = 'state'

for n in range(1,4):

    success = pd.concat([success, pd.DataFrame({ str(n) : kickstarter.loc[kickstarter['duration_in_month']==n,:]['state'].value_counts(normalize=True)})], axis=1, sort=False)



ax=sns.heatmap(success, cmap='Blues' ,linecolor='k', linewidths=0.2)

ax.set_title('State distribution by project duration (in months)',size=16, weight='bold')

plt.show()
mask = (success.index == 'successful') | (success.index == 'failed')

success_or_fail = success.loc[mask,:]



success_or_fail
ax = plt.gca()



success_or_fail.T.plot.bar(width=0.35 ,stacked=True, figsize=(8,7), color=['green','red'], ax=ax)

ttl = ax.set_title('Success & Failed rates by project duration (in months)',size=16, weight='bold')

ttl.set_position([0.5,1.05])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

plt.show()
plt.figure(figsize=(18,9))

plt.xticks(np.arange(1, 93, 3))

duration_success = kickstarter.groupby('project_duration')['is_success'].mean()

duration_success.index = duration_success.index.days

ax = sns.lineplot(data=duration_success)



ax.set_xlabel('Duration of Project (in days)', size=10 , weight='semibold')

ax.set_ylabel('Success rate', size=10 , weight='semibold')

ax.set_title('Success rate by by project duration in days',size=16,weight='bold')



plt.show()
plt.figure(figsize=(12,6))

pledged_in_months = kickstarter.groupby(['opening_month','duration_in_month'])['pledged_of_goal_usd'].agg('median').unstack().round(2)



ax = sns.barplot(data=pledged_in_months)



#sns.set(style="whitegrid")

ax.set_xlabel('Duration of Project (in months)', size=10 , weight='semibold')

ax.set_ylabel('MEDIAN - % Pledged out of Goal', size=10 , weight='semibold')

ax.set_title('% pledged of goal by projects duration in monthes ',size=16,weight='bold')



plt.show()
plt.figure(figsize=(12,6))

ax=kickstarter.groupby('opening_month')['usd_pledged_real'].mean().plot.bar(color=sns.color_palette("Set1", desat=0.5))

ax.set_title('Average money pledged by the opening month of the project',size=16, weight='bold')

ax.set_xlabel('Opening month')

ax.set_ylabel('Average money pledged')

plt.show()
plt.figure(figsize=(12,6))

ax=kickstarter.groupby('opening_month')['is_success'].mean().plot.bar()

ax.set_title('Average success rate by the opening month of the project',size=16, weight='bold')

plt.show()
plt.figure(figsize=(10,15))

under_pledged = kickstarter[kickstarter['pledged_of_goal_usd']<3]

ax = sns.violinplot(x="opening_month", y="pledged_of_goal_usd", data=under_pledged)

ax.set_xlabel('Opening month of the Project', size=10 , weight='semibold')

ax.set_ylabel('MEAN % Pledged out of Goal', size=10 , weight='semibold')

ax.set_title('% pledged out of goal by duration across opening months of projects',size=16, weight='bold')

plt.show()
plt.figure(figsize=(14,7))

plt.xticks(np.arange(1, 13, 1))

pledged_in_months = kickstarter.groupby(['opening_month','duration_in_month'])['pledged_of_goal_usd'].mean().unstack().round(2)

pledged_in_months.rename(columns={0:'Less than a month',

                          1:'Between 1 & 2 months',

                          2:'Between 2 & 3 months',

                          3:'More than 3 months'}, 

                          inplace=True)

ax = sns.lineplot(data=pledged_in_months)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

ax.set_xlabel('Opening month of the Project', size=10 , weight='semibold')

ax.set_ylabel('MEAN - % Pledged out of Goal', size=10 , weight='semibold')

ax.set_title('% pledged out of the goal by duration across opening months of projects',size=16,weight='bold')

ax.set_ylim(bottom=0)



plt.show()