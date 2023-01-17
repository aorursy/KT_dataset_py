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
data_dir = '../input/'
Seeds = pd.read_csv(data_dir +'WNCAATourneySeeds.csv')
CompactResults= pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')
#CompactResults= pd.read_csv(data_dir + 'WRegularSeasonCompactResults.csv')
Seeds.head()
CompactResults.head()
def seed_int(x):
    temp = int(x[1:3])
    return temp
Seeds['tier'] = Seeds.Seed.apply(seed_int)

def seed_region(x):
    temp=x[:1]
    return temp
Seeds['region']=Seeds.Seed.apply(seed_region)
Seeds.head()
tempSeeds=Seeds.loc[:,['Season','TeamID','tier','region']]
tempSeeds.columns=['Season','WTeamID','WTier','WRegion']
temp=CompactResults.merge(tempSeeds,how='left',on=['Season','WTeamID'])
temp.head()
tempSeeds.columns=['Season','LTeamID','LTier','LRegion']
merge=temp.merge(tempSeeds,how='left',on=['Season','LTeamID'])
merge.head()
#merge['history']=(merge['Season']-1990)
merge['tierDiff']=(merge['WTier']-merge['LTier'])
merge=merge.loc[:,['WLoc','WRegion','LRegion','tierDiff']]
merge.head()
merge['VSRegion']=merge['WRegion']+merge['LRegion']
merge.head()
merge=merge.loc[:,['WLoc','tierDiff','VSRegion']]
merge.head()
mergeReverse=merge

def reverse(x):
    temp1 = x[:1]
    temp2 = x[1:2]
    temp=temp2+temp1
    return temp

mergeReverse['VSRegion2']=merge['VSRegion'].apply(reverse)
mergeReverse['tierDiff']=-mergeReverse['tierDiff']

mergeReverse.head()
def reverseLoc(x):
    if x == 'H':
       return 'A' 
    elif x == 'A':
        return 'H'
    else:
        return x

mergeReverse['WLoc2']=merge['WLoc'].apply(reverseLoc)

mergeReverse.head()
mergeReverse=mergeReverse.loc[:,['WLoc2','tierDiff','VSRegion2']]
mergeReverse.head()
mergeReverse.columns=['WLoc','tierDiff','VSRegion']
mergeReverse['feature']=1
mergeReverse.head()
merge=merge.loc[:,['WLoc','tierDiff','VSRegion']]
merge['feature']=0
merge.head()
#test append
temp1=merge.head()
temp2=mergeReverse.head()
temp1.append(temp2,ignore_index=True)

merge=merge.append(mergeReverse,ignore_index=True)
merge.head()
tempdummies = pd.get_dummies(merge['WLoc'],prefix='WLoc')
tempdummies.head()
merge=pd.concat([merge,tempdummies],axis=1)
merge.head()
merge=merge.drop(labels=['WLoc'],axis=1)
merge.head()
tempdummies = pd.get_dummies(merge['VSRegion'],prefix='VSRegion')
tempdummies.head()
merge=pd.concat([merge,tempdummies],axis=1)
merge.head()
merge=merge.drop(labels=['VSRegion'],axis=1)
merge.head()
#from sklearn.utils import shuffle
#merge=shuffle(merge)
#merge.head()
#scaler???
feature=merge.feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#where to find data about WLoc in submission
merge=merge.drop(labels=['WLoc_A','WLoc_H','WLoc_N'],axis=1)
merge.head()
train=merge.drop(labels=['feature'],axis=1)
train.head()
hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5,7,10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [3,5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}

clf = RandomForestClassifier(random_state=3)
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)

all_X = train.values
all_y = feature

grid.fit(all_X, all_y)
best_params = grid.best_params_
best_score = grid.best_score_
best_score
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,all_X, all_y,cv=10)
scores.mean()

Submission = pd.read_csv(data_dir + 'WSampleSubmissionStage1.csv')
output=Submission
output.head()
def year(x):
    temp = (x[:4])
    return temp
output['year'] = output.ID.apply(year)
output.head()
def team(x):
    temp = (x[5:9])
    return temp
output['team1'] = output.ID.apply(team)
output.head()
def team(x):
    temp = (x[10:14])
    return temp
output['team2'] = output.ID.apply(team)
output.head()
Seeds.head()
tempSeeds=Seeds.loc[:,['Season','TeamID','tier','region']]
tempSeeds.head()
tempSeeds.columns=['year','team1','tier1','region1']
tempSeeds.head()
output.year=output.year.astype(int)
output.team1=output.team1.astype(int)
output.team2=output.team2.astype(int)
output.dtypes
tempSeeds.dtypes
temp=pd.merge(output,tempSeeds,how='left',on=['year','team1'])
temp.head()
tempSeeds.columns=['year','team2','tier2','region2']
tempSeeds.head()
output=pd.merge(temp,tempSeeds,how='left',on=['year','team2'])
output.head()
#output['history']=(output['year']-1990)
#output.head()
output['tierDiff']=(output['tier1']-output['tier2'])
output.head()
output['VSRegion']=output['region1']+output['region2']
output.head()
train.columns
output=output.loc[:,['ID', 'tierDiff','region1','region2','VSRegion']]
output.head()
tempdummies = pd.get_dummies(output['VSRegion'],prefix='VSRegion')
tempdummies.head()
output=pd.concat([output,tempdummies],axis=1)
output.head()
output.columns
merge.columns
columns=[ 'tierDiff',
       'VSRegion_WW', 'VSRegion_WX', 'VSRegion_WY', 'VSRegion_WZ',
       'VSRegion_XW', 'VSRegion_XX', 'VSRegion_XY', 'VSRegion_XZ',
       'VSRegion_YW', 'VSRegion_YX', 'VSRegion_YY', 'VSRegion_YZ',
       'VSRegion_ZW', 'VSRegion_ZX', 'VSRegion_ZY', 'VSRegion_ZZ']

best_rf = grid.best_estimator_
outcome= best_rf.predict(output[columns])
proba= best_rf.predict_proba(output[columns])
proba
Submission = pd.read_csv(data_dir + 'WSampleSubmissionStage1.csv')
Submission.head()
Submission.Pred=proba[:,1]

Submission.head()
Submission.to_csv('prediction1.csv', index=False)