# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Regular Matches

Regular=pd.read_csv('../input/regularseasoncompactresults/WRegularSeasonCompactResults.csv')
Regular.head()
#list all teams

w1=Regular['WTeamID']

w2=Regular['LTeamID']

all_set=set(pd.concat([w1,w2],axis=0))
# Establish Elo rating system 

elom=1500

elow=400

kfactor=64



#Find the probability of A wins, expected score

def prob(A,B):

    #A is elo rating for A before the match

    p= 1/(1+(10**((B-A)/400)))

    return p

def update(rating, score, expect):

    return rating+ kfactor*(score-expect)
#Accumulate ratings for each team

elorating={

    'TeamID':list(all_set),

    'Rating':[1500 for i in range(len(all_set))]

}



Elorating=pd.DataFrame(elorating)

Elorating.head()
Team_Elo=dict(zip(Elorating['TeamID'],Elorating['Rating']))
#input original elo ratings and update ratings

def update_season(season,original):

    if season==2019:

        return original

    #Filter Dataframe by season

    match_i=Regular[Regular['Season']==season]

    #Create a match list in this season

    matches=zip(match_i['WTeamID'],match_i['LTeamID'])

    match_list=list(matches)

    copy=original

    for i in match_list:

        A=i[0]

        B=i[1]

        #Expected score for win and lose team before matches

        ExpectA=prob(copy[A],copy[B])

        ExpectB=prob(copy[B],copy[A])

        #get new elo ratings

        NewA=update(copy[A], 1, ExpectA)

        NewB=update(copy[B], 0, ExpectB)

        #update information for each matches

        copy[A]=NewA

        copy[B]=NewB

    Elorating['%s' %season]=copy.values()

    return update_season(season+1,copy)
#Get Elo ratings for 2018

elo2018=update_season(1998,Team_Elo)
Elorating.head()
sample=Elorating.sample(5)

sample.head()
X=[i for i in range(1998,2019)]

y=sample.iloc[0,2:]

fig=plt.figure(figsize=(10,10))

sns.lineplot(x=X,y=sample.iloc[0,2:],label='%s'%list(sample['TeamID'])[0])

sns.lineplot(x=X,y=sample.iloc[1,2:],label='%s'%list(sample['TeamID'])[1])

sns.lineplot(x=X,y=sample.iloc[2,2:],label='%s'%list(sample['TeamID'])[2])

sns.lineplot(x=X,y=sample.iloc[3,2:],label='%s'%list(sample['TeamID'])[3])

sns.lineplot(x=X,y=sample.iloc[4,2:],label='%s'%list(sample['TeamID'])[4])

plt.xlabel('Season')

plt.ylabel('Elo Rating')

plt.xticks(X)

plt.title('Elo rating for samples')
#import submission stage 2

submission2=pd.read_csv('../input/submission/WSampleSubmissionStage2 (1).csv',sep='_|,',engine='python')


submission2=submission2.drop('Prediction',axis=1)
matches=zip(submission2['TeamA'],submission2['TeamB'])

match_list=list(matches)

prediction=[]

for i in match_list:

    A=i[0]

    B=i[1]

    #Expected score for win and lose team before matches

    ExpectA=prob(elo2018[A],elo2018[B])

    prediction.append(ExpectA)
submission2['prediction']=prediction
submission2