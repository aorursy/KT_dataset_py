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
test = pd.read_csv('../input/SampleSubmission.csv')

RegularSeasonCompactResults = pd.read_csv('../input/RegularSeasonCompactResults.csv')

TourneyCompactResults = pd.read_csv('../input/TourneyCompactResults.csv')
train = pd.DataFrame()
Teams = pd.read_csv('../input/Teams.csv')
matches = []

other_teams = Teams['Team_Id'].values

for team1 in Teams['Team_Id']:

    other_teams = np.delete(other_teams, 0)

    for team2 in other_teams:

        matches.append([team1,team2])

            

lst1 = [item[0] for item in matches]

train['team1'] = lst1

lst2 = [item[1] for item in matches]

train['team2'] = lst2



train.tail(20)


RegularSeasonCompactResults['seasonMatches'] = RegularSeasonCompactResults[['Wteam','Lteam']].values.tolist()
RegularSeasonCompactResults.head()
train['matches'] = train[['team1', 'team2']].values.tolist()



RegularSeasonCompactResults['Result'] = RegularSeasonCompactResults['seasonMatches'].apply(lambda x : 0 if x[0]>x[1] else 1 )        

RegularSeasonCompactResults['seasonMatchesSorted'] = RegularSeasonCompactResults['seasonMatches'].apply(lambda x : [x[1],x[0]] if x[0] > x[1] else x)        
RegularSeasonCompactResults['seasonMatchesSorted'] = RegularSeasonCompactResults['seasonMatchesSorted'].apply(tuple)

RegularSeasonCompactResults['seasonMatchesGrouped'] = RegularSeasonCompactResults.groupby("seasonMatchesSorted")["Result"].transform('mean')
RegularSeasonCompactResults = RegularSeasonCompactResults.rename(columns={'seasonMatchesGrouped': 'ResultsGrouped'})

train['matches']=train['matches'].apply(tuple)

def result(x):

    a = RegularSeasonCompactResults.loc[(RegularSeasonCompactResults['seasonMatchesSorted']==x)]['ResultsGrouped'].values

    return a[0]

seasonMatchesSorted = RegularSeasonCompactResults['seasonMatchesSorted']

train['result'] = train['matches'].apply(lambda x : result(x) if x in seasonMatchesSorted.tolist()   else 0.5)    
Y = train.result

X = train.drop(['result','matches'], axis=1)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

clf2 = LinearRegression()

clf2.fit(train_X, train_y)

print("LinearRegression")

print(clf2.predict(val_X),'\n')

model_train_pred = clf2.predict(val_X)

    

sub = pd.read_csv('../input/SampleSubmission.csv')

sub["team1"] = sub["Id"].apply(lambda x: int(x.split("_")[1]))

sub["team2"] = sub["Id"].apply(lambda x: int(x.split("_")[2]))

sub.head()
sub_X = sub.drop(['Id','Pred'], axis=1)
sub["Pred"] =  clf2.predict(sub_X)

sub = sub[["Id", "Pred"]]

sub.head(30)
sub.to_csv("CF.csv", index=False)
