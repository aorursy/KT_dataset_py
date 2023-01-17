import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

print ('Done.')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Read in our datasets

draft78 = pd.read_csv('../input/draft78.csv')

season78 = pd.read_csv('../input/season78.csv')
#Focus on players drafted in Round 1 or Round 2

draft78_top = draft78[draft78['Pick']<60]
#Visualize longevity of a given position in the draft

longevity = draft78_top.groupby(['Pick'])['Yrs'].mean()

longevity = longevity.reset_index()

ax = longevity.plot(x='Pick', y='Yrs', kind='line', figsize=(17,8), color='b');

plt.ylabel('Average Longevity');

plt.xlabel('Draft Number');

plt.ylim((1,12))

plt.xlim((0,60))



#Add general regression line

x = longevity['Pick']

y = longevity['Yrs']

z = np.polyfit(x, y, 1)

p = np.poly1d(z)

plt.plot(x,p(x),"b--")



#Get details about the regression

import statsmodels.api as sm

x = sm.add_constant(x)

est = sm.OLS(y, x).fit()

est.summary()
#Merge winshares and draft status (for those drafted in the first two rounds)

draft_merge = pd.merge(draft78_top, season78, on='Player', how='inner')

draft_merge[:5]
#Let's visualize all the #1 picks by winshares over their career

firstpicks = draft_merge[draft_merge['Pick']==1].groupby(['Player']).count()

#Organize by longevity and convert to a list

firstpicks = firstpicks.sort_values(by='Yrs', ascending=False)

first_list = firstpicks.index.tolist()
#Set the number of graphs in the facet chart

graphs = len(first_list)



#create a list of positions for the chart

position = []

for i in range(6):

    for j in range(6):

        b = i,j

        position.append(b)



#Create base of subplot chart.. rows x columbs = graphs

fig, axes = plt.subplots(nrows=6, ncols=6, sharey=False, sharex=False, figsize=(12,12))

fig.subplots_adjust(hspace=.5)



#Fill in base with graphs based off of position

for i in range(graphs):

    draft_merge[draft_merge['Player']==first_list[i]].plot(kind='bar',x='Season', y='WS', 

                                                           ax=axes[position[i]], legend=False)



#Set the formatting elements of the axes for each graph

for i in range(graphs):

    axes[position[i]].set_title(first_list[i], size = 6)

    axes[position[i]].tick_params(labelsize=5)

    axes[position[i]].set_xlabel("Year", size = 5)

    axes[position[i]].set_ylim((0,20))
#What's up with Patrick Ewing?

season78[season78['Player']=='Patrick Ewing']
#Look at the relationship between longevity and average WS

year_group = draft_merge.groupby(['Player','Yrs'])['WS'].agg(['mean','count'])

year_group = year_group.reset_index()



#Let's see if there are years were WS is not counted 

year_group['count_check']=year_group['Yrs']-year_group['count']

weirdos = year_group[year_group.count_check != 0]



#Okay, so some of these are people who have the same names, what about the others

duplicates = weirdos[weirdos.duplicated('Player') == True]['Player'].tolist()

true_weirdos = weirdos[weirdos['Player'].isin(duplicates) == False]

true_weirdos.sort_values('count_check')



#Very strange,let's get them out of the dataset

year_group = year_group[year_group['Player'].isin(true_weirdos['Player'].tolist()) == False]



#And rename the column while we're at it

year_group.columns.values[2] = 'Avg_WS'

year_group
#Now let's graph longevity/Avg_WS

year_group_graph = year_group.iloc[:,0:3]

year_group_graph = year_group_graph.set_index('Player')



fig, ax = plt.subplots() 

year_group_graph.plot(kind='scatter', x='Yrs', y='Avg_WS', figsize=(12,6), ax=ax)



for index, rows in year_group_graph.iterrows():

    ax.annotate(index, rows)