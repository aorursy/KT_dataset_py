# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

sns.set(style="white", color_codes=True)



#now reading the csv files

d=pd.read_csv("../input/deliveries.csv")

k=pd.read_csv("../input/matches.csv")





#we want to label the bars later           #autolabel code :http://composition.al/blog/2015/11/29/a-better-way-to-add-labels-to-bar-charts-with-matplotlib/

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%d' % int(height),ha='center', va='bottom')
###Six heatmap

six = d[ d.batsman_runs == 6 ] #Considering only those deliveries where 6 have been hit

six = six[['over', 'batting_team','batsman_runs']]  #This three columns is what we require

result = pd.pivot_table(six, index='batting_team', columns='over', values='batsman_runs', aggfunc=np.sum) #creating a pivot table for applying heat map

sns.heatmap(result, annot=False, fmt="g" , cmap='viridis')  #heat map

plt.show()

#to know more about heat map : http://stackoverflow.com/questions/37790429/seaborn-heatmap-using-pandas-dataframe
### Finding the  best strike rate in death overs.Death overs = over>15



death_df = d[ d.over > 16 ]  #Taking data for overs > 16

def balls_faced(x):

    return len(x)



temp_df = death_df.groupby('batsman')['batsman_runs'].agg([balls_faced,'sum']).reset_index() #making new column 'ball faced' and 'total run'

temp_df = temp_df.ix[temp_df.balls_faced>50,:]   #we will consider only the player who have faced more than 50 balls

temp_df['strike-rate in death overs'] = (temp_df['sum'] / temp_df['balls_faced'])*100   #adding our required column 'strike-rate'

temp_df = temp_df.sort_values(by='strike-rate in death overs', ascending=False).reset_index(drop=True)   #now we sort them by their strike rates

temp_df = temp_df.iloc[:10,:]     #lets select the top 10 players

temp_df = temp_df[['batsman','strike-rate in death overs']]

temp_df
###Now bar diagram to represent it



labels = np.array(temp_df['batsman'])

ind = np.arange(len(labels))

width = 0.8

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_df['strike-rate in death overs']), width=width, color='m')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("strike-rate in death overs")

ax.set_title("Best finishers!")

autolabel(rects)

plt.show()
###Lets find the team which have used thir powerplays the best



p_df = d[ d.over <= 6 ]  #Taking data for overs <= 6

def over_played(x):

    return (len(x)//6)

    

power_df = p_df.groupby('batting_team')['batsman_runs'].agg([over_played, 'sum']).reset_index() #making new column 'ball faced' and 'total run'

power_df = power_df.ix[power_df.over_played>100,:]   #we will consider only the team who have faced more than 100 overs

power_df['run-rate'] =(power_df['sum'] /power_df['over_played'])   #adding our required column 'run-rate'

power_df = power_df.sort_values(by='run-rate', ascending=False).reset_index(drop=True)  #now we sort them by their run-rate

power_df = power_df[['batting_team','over_played','run-rate']]

power_df


k=pd.read_csv("../input/matches.csv")

###Finding number of match played each year

def tot_match(x) :

    return (x>=0).sum()



totmatch = k.groupby('season')['win_by_runs'].agg(tot_match)

mtch = []

for i in range(2008,2017) :

    mtch.append(totmatch[i])



###Finding avg first innings score

score_df = d[d.inning == 1]

score_df = score_df.groupby('match_id')['batsman_runs'].agg('sum')



tot = 0

avg = []

k = 0

j = 0

for i in range(1,578) :

    tot = tot + score_df[i]

    if (i-k) >= mtch[j] :

        j += 1

        k = k + mtch[j-1]

        avg.append(tot/mtch[j-1])

        tot = 0

print('Season' + '  ' + 'Total matches' + '  ' + 'Avg First Innings Score')

for i in range(9) :

    ans =  '  '+ str(2008+i) + '       ' +str(mtch[i]) + '           '  + str(avg[i])

    print(ans)