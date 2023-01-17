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





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style



style.use('fivethirtyeight')





matches=pd.read_csv('../input/matches.csv')

deliveries=pd.read_csv('../input/deliveries.csv')

#Batsmen runs:



batsmen = deliveries.groupby(["match_id", "inning", "batting_team", "batsman"])

batsmen = batsmen["batsman_runs"].sum().reset_index()

print (batsmen.head(5))
#dismissals



dismissals = deliveries[pd.notnull(deliveries['player_dismissed'])]

dismissals = dismissals[['match_id','inning','player_dismissed','dismissal_kind','fielder']]

dismissals.rename(columns={'player_dismissed':'batsman'},inplace=True)



batsmen=batsmen.merge(dismissals,left_on=['match_id','inning','batsman'],right_on=['match_id','inning','batsman'], how= "left")



batsmen['dismissal_kind']=batsmen.dismissal_kind.fillna('not_out')



batsmen['fielder']=batsmen.fielder.fillna('-')



print (batsmen.head(5))
#Number of innings

no_of_innings=batsmen.groupby(['inning','batsman']).size().reset_index()

no_of_innings=no_of_innings.groupby(['batsman']).sum().reset_index()

no_of_innings=no_of_innings.drop(['inning'],1)

no_of_innings.rename(columns={0:'no_of_innings'},inplace=True)



print (no_of_innings.head(5))
#dismissal_types

dismissal_group=batsmen[['dismissal_kind','batsman']]

dismissal_group=dismissal_group.groupby(['batsman','dismissal_kind']).size().reset_index()

dismissal_group.rename(columns={'0':'No_of_times'},inplace=True)

not_outs=dismissal_group[dismissal_group['dismissal_kind']=='not_out']

not_outs=not_outs.drop(['dismissal_kind'],1)

not_outs.rename(columns={0:'no_of_not_outs'},inplace=True)

print (not_outs.head(5))
#batting_average= Runs/innings-not_outs



total_runs=batsmen.groupby(['batsman'] ).sum().reset_index()



total_runs=total_runs.drop(['inning','match_id'],1)

batsmen_overall=no_of_innings.merge(total_runs,on='batsman')

batsmen_overall=batsmen_overall.merge(not_outs,on='batsman')

batsmen_overall['batting_average']=batsmen_overall['batsman_runs']/(batsmen_overall['no_of_innings']-batsmen_overall['no_of_not_outs'])

batsmen_overall=batsmen_overall.replace([np.inf, -np.inf], 0)

batsmen_overall.rename(columns={'batsman':'player'},inplace=True)



A=list(range(len(batsmen_overall)))

plt.scatter(A,batsmen_overall['batting_average'])

plt.xlabel('Player')

plt.ylabel('Average')

plt.show()



#bowlers

bowlers=deliveries[['ball','bowler','extra_runs','total_runs','player_dismissed','dismissal_kind','fielder']]

#Overs

total_overs=bowlers.groupby(['bowler','ball']).size().reset_index()

total_overs.rename(columns={0:'no_of_each_balls'},inplace=True)

total_overs['total_balls']=0

   

for i in range(0,len(total_overs)):

	if total_overs.ball.iloc[i]==1:

		total_overs['total_balls'].iloc[i] = total_overs['no_of_each_balls'].iloc[i]+total_overs['no_of_each_balls'].iloc[i+1]+total_overs['no_of_each_balls'].iloc[i+2]+total_overs['no_of_each_balls'].iloc[i+3]+total_overs['no_of_each_balls'].iloc[i+4]+total_overs['no_of_each_balls'].iloc[i+5]

        
#runs conceded



runs_conceded=bowlers.groupby(['bowler','total_runs']).size().reset_index()

runs_conceded.rename(columns={0:'runs_in_different_ways'},inplace=True)

runs_conceded['total_runs_in_different_ways']=0

runs_conceded['total_runs_in_different_ways']=runs_conceded['runs_in_different_ways']*runs_conceded['total_runs']

runs_conceded=runs_conceded[['bowler','total_runs_in_different_ways']]

runs_conceded=runs_conceded.groupby(['bowler']).sum().reset_index()
#wickets



wickets=bowlers[['bowler','dismissal_kind']]

wickets=wickets.dropna()



wickets=wickets[wickets.dismissal_kind != 'run out']

wickets=wickets.groupby(['bowler']).size().reset_index()

wickets.rename(columns={0:'total_wickets'},inplace=True)
#Bowling average=runs/wicket taken



Bowling_average=runs_conceded.merge(wickets,on='bowler',how='left')

Bowling_average['bowling_average']=Bowling_average['total_runs_in_different_ways']/Bowling_average['total_wickets']

#economy=runs/overs   

Economy=runs_conceded.merge(total_overs,on='bowler',how='left')

Economy=Economy[Economy.total_balls != 0]

Economy=Economy[['bowler','total_balls','total_runs_in_different_ways']]

Economy['total_overs']=Economy['total_balls']/6

Economy['economy']=Economy['total_runs_in_different_ways']/Economy['total_overs']
#player,runs,batting_average,wickets,bowling_average,economy



bowler_overall=wickets.merge(Bowling_average[['bowler','total_runs_in_different_ways','bowling_average']], on='bowler',how='left')

bowler_overall=bowler_overall.merge(Economy[['bowler','total_balls','economy','total_overs']],on='bowler',how='left')

bowler_overall.rename(columns={'bowler':'player'},inplace=True)



player_overall=batsmen_overall.merge(bowler_overall,on='player',how='left')



print(player_overall.head(5))

df= player_overall 

df.fillna(-9999, inplace= True)

from sklearn import preprocessing, cross_validation, neighbors

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
#Batting Average Prediction



X1 = df.drop(['player','batting_average','total_wickets','total_runs_in_different_ways','bowling_average','total_balls','economy','total_overs'],1)

y1 = np.asarray(df['batting_average'], dtype=np.float)

X1_train, X1_test, y1_train, y1_test = cross_validation.train_test_split(X1,y1,test_size=0.2)

#MS Dhoni's statistics and his average = 37.88

example_measures = np.array([143,3561,49])

example_measures = example_measures.reshape(1,-1)

#KNN

n_neighbors = 2

clf1 = neighbors.KNeighborsRegressor(n_neighbors)

clf1.fit(X1_train.values, y1_train)

accuracy1= clf1.score(X1_test,y1_test)

print(accuracy1)

prediction = clf1.predict(example_measures)

print(prediction) 
#Linear

clf=LinearRegression()

clf.fit(X1_train.values, y1_train)

accuracy= clf.score(X1_test,y1_test)

print(accuracy)

prediction = clf.predict(example_measures)

print(prediction) 
#SVM

clf = svm.SVR(kernel='linear')

clf.fit(X1_train.values, y1_train)

accuracy= clf.score(X1_test,y1_test)

print(accuracy)

prediction = clf.predict(example_measures)

print(prediction) 