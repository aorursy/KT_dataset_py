#import all the necessary libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
fifa = pd.read_csv("../input/data.csv",index_col=0)
fifa.head()
top10_nation_count = fifa.groupby('Nationality').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,8))

ax=top10_nation_count.plot.bar(rot=60)

plt.ylabel("Number of players")

plt.xlabel("Countries")

plt.title("Top 10 Countries based on player count")

ax.set_ylim(0,1800)



for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+20, str(i.get_height()),fontsize=11)
fifa.sort_values(by='Overall',ascending=False).head(10)[['Name','Nationality','Club','Overall']]
per_nation = fifa.groupby('Nationality').size()

plt.figure(figsize=(10,8))

ax=sns.countplot(per_nation[per_nation<=10])

plt.xlabel("Number of Players")

plt.ylabel("Number of Countries")

plt.title("Number of Countries with least player counts")

ax.set_ylim(0,28)



for i in ax.patches:

    ax.text(i.get_x()+0.3, i.get_height()+0.5, str(i.get_height()),fontsize=11)
top10=fifa.groupby('Club')['Overall'].mean().sort_values(ascending=False).head(10)

top10_data = fifa.loc[fifa['Club'].isin(top10.index) & fifa['Age']]



plt.figure(1,figsize=(10,6))

sns.violinplot(data =top10_data, x='Club',y='Age')

plt.xticks(rotation=45)

plt.title("Distribution of Players Age in Top 10 Clubs")
ctry_count = fifa.groupby('Nationality').size()

ctry_count_list = ctry_count[ctry_count>100].sort_values(ascending=False).index



df = fifa[fifa['Nationality'].isin(ctry_count_list)]



top10_ctry = df.groupby('Nationality')['Overall'].mean().sort_values(ascending=False).head(10)

top1_ctry_data = df.loc[df['Nationality'].isin(top10_ctry.index) & df['Age']]

top1_ctry_data



plt.figure(1,figsize=(10,6))

sns.violinplot(data=top1_ctry_data,x='Nationality',y='Age')

plt.xticks(rotation=45)

plt.title("Distribution of Players`s Age in Top 10 Countries")
fifa['Wage']=fifa['Wage'].str.extract('(\d+)',expand=True).astype(int)
top_wages_club = fifa.groupby('Club')['Wage'].agg(max).sort_values(ascending=False).head(10)

ax=top_wages_club.plot.bar(title="Club's Highest Wage",figsize=(10,8),rot=70)

plt.ylabel('Wages (Thousand Euro)')

ax.set_ylim(0,600)



for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+10, str(i.get_height()),fontsize=11)
plt.figure(figsize=(15,10))

sns.scatterplot(data=fifa,y='Wage',x='Overall',hue='Age',size='Age',sizes=(20, 200),palette="Set1")

plt.title("Distribution of wages based on the Players Age and Overall Performance")

plt.xlabel("Overall Scores")

plt.ylabel("Wages (Thousand Euro)")
#Take the mean of Age and Wage

mean_age = fifa['Age'].mean()

mean_wage = fifa['Wage'].mean()



less_mean_count = fifa[fifa['Age'] < mean_age]['Age'].count()

more_mean_count = fifa[fifa['Age'] >= mean_age]['Age'].count()



labels_Age = ['% of Players below Mean Age', '% of Players above Mean Age']

sizes_Age = [less_mean_count, more_mean_count]





less_mean_wage = fifa[fifa['Wage'] < mean_wage]['Wage'].count()

more_mean_wage = fifa[fifa['Wage'] >= mean_wage]['Wage'].count()



labels_Wage = ['% of Players below Mean Wage', '% of Players above Mean Wage']

sizes_Wage = [less_mean_wage, more_mean_wage]

colors = ['yellowgreen','lightcoral']





fig,axarr = plt.subplots(1,2, figsize=(16,8))



axarr[0].pie(x=sizes_Age,labels=labels_Age,autopct='%1.1f%%'),

axarr[1].pie(x=sizes_Wage,labels=labels_Wage,autopct='%1.1f%%',colors=colors)

axarr[0].set_title('Distribution of Players with respect to Mean Age')

axarr[1].set_title('Distribution of Players with respect to Mean Wage')

plt.axis('equal')



greater_mean_age_df = fifa[fifa['Age'] >= mean_age]

lesser_mean_age_df = fifa[fifa['Age'] < mean_age]



less_wage_more_age = greater_mean_age_df[greater_mean_age_df['Wage'] < mean_wage]['Wage'].count()

more_wage_more_age = greater_mean_age_df[greater_mean_age_df['Wage'] >= mean_wage]['Wage'].count()



less_wage_less_Age = lesser_mean_age_df[lesser_mean_age_df['Wage'] < mean_wage]['Wage'].count()

more_wage_less_Age = lesser_mean_age_df[lesser_mean_age_df['Wage'] >= mean_wage]['Wage'].count()



labels_more_Age = ['% of Players below Mean Wage', '% of Players above Mean Wage']

sizes_more_Age = [less_wage_more_age, more_wage_more_age]



labels_less_Age = ['% of Players below Mean Wage', '% of Players above Mean Wage']

sizes_less_Age = [less_wage_less_Age, more_wage_less_Age]



fig,axarr = plt.subplots(1,2, figsize=(16,8))

axarr[0].pie(x=sizes_more_Age,labels=labels_more_Age,autopct='%1.1f%%')

axarr[1].pie(x=sizes_less_Age,labels=labels_less_Age,autopct='%1.1f%%')

plt.axis('equal')

axarr[0].set_title('Distribution of Players w.r.t Wages and Age>=Mean Age')

axarr[1].set_title('Distribution of Players w.r.t Wages and Age<Mean Age')



#Create a Dataframe with 2 columns such as Number of players and Parameters.



Parameters = ['Total Players','< Mean Age', '>= Mean Age', '< Mean Wage', '>= Mean Wage' , '< M.age & < M.Wage','< M.age & >= M.Wage','>= M.age & < M.Wage','>= M.age & >= M.Wage'  ]

Number_of_Players = [len(fifa),less_mean_count,more_mean_count,less_mean_wage,more_mean_wage,less_wage_less_Age,more_wage_less_Age,less_wage_more_age,more_wage_more_age]

d = {'Parameters':Parameters,

    'Number of Players':Number_of_Players

    }

players_stats = pd.DataFrame(d)
plt.figure()

ax=players_stats.plot.bar('Parameters','Number of Players',title='Number of Players w.r.t selected Parametes',rot=45,figsize=(14,10),legend=False)

plt.ylabel("Number of Players")

ax.set_ylim(0,20000)

for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+200, str(i.get_height()),fontsize=11)
fifa_copy = fifa.copy()

fifa_copy['Difference of Points'] = fifa_copy['Potential']-fifa_copy['Overall']

fifa_copy.sort_values(by='Difference of Points', ascending = False).head(10)[['Name','Nationality','Overall','Potential']]

sorted_Skills_reverse = fifa.sort_values(by=['Skill Moves','Wage'],ascending=[True,False])[['ID','Name','Club','Nationality','Wage','Skill Moves']]

sorted_Skills_reverse.head(10)
sorted_Skills = fifa[fifa['Club'].notnull()].sort_values(by=['Skill Moves','Wage'],ascending=[False,True])[['ID','Name','Club','Nationality','Wage','Skill Moves']]

sorted_Skills.head(10)
plt.figure(figsize=(15,10))

sns.scatterplot(data=fifa,y='Wage',x='Age',hue='Work Rate')

plt.title("Players wages vs Players Age with respect to work rate")

plt.xlabel("Players Age")

plt.ylabel("Wages  (Thousand Euro)")
g=sns.FacetGrid(fifa, col='Preferred Foot')

g.map(sns.kdeplot,"Overall")