# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sq
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#just reading data
con = sq.connect("../input/database.sqlite")
team_att = pd.read_sql_query("SELECT * from Team_Attributes", con)
team = pd.read_sql_query("SELECT * from Team", con)
match = pd.read_sql_query("SELECT * from Match", con)
match = match[['date', 'home_team_goal', 'away_team_goal', 'home_team_api_id', 'away_team_api_id', 
              'goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession',
              'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
              'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD',
              'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']]
print("sahin")
#shuffle match rows so split tables are randomized
match = match.reindex(np.random.permutation(match.index))


#split match data into training, validation, and test sets
m_train = match.iloc[:17861]
m_test = match.iloc[17861:21108]

m1=match.iloc[0:21109]
m_valid = match.iloc[21108:]
print("sahin")

l =list(range(21109))
for i in range (0,21109):
    if(match.loc[i].home_team_goal>match.loc[i].away_team_goal):
        l[i]=1
        
        
    elif(match.loc[i].home_team_goal<match.loc[i].away_team_goal):
        l[i]=2
    elif(match.loc[i].home_team_goal==match.loc[i].away_team_goal):
        l[i]=0
l_train=l[:17861]
l_test=l[17861:21108]
print("sahin")
print(l)

print(l_train)

print(l_test)
 
m_tr=m_train.drop(['date'],1)
m_tr=m_tr.drop(['BSD'],1)
m_tr=m_tr.drop(['BSA'],1)
m_tr=m_tr.drop(['shoton'],1)
m_tr=m_tr.drop(['shotoff'],1)
m_tr=m_tr.drop(['foulcommit'],1)
m_tr=m_tr.drop(['card'],1)
m_tr=m_tr.drop(['cross'],1)
m_tr=m_tr.drop(['corner'],1)
m_tr=m_tr.drop(['possession'],1)
m_tr=m_tr.drop(['B365H'],1)
m_tr=m_tr.drop(['B365D'],1)
m_tr=m_tr.drop(['B365A'],1)
m_tr=m_tr.drop(['PSA'],1)
m_tr=m_tr.drop(['WHH'],1)
m_tr=m_tr.drop(['WHA'],1)
m_tr=m_tr.drop(['SJH'],1)
m_tr=m_tr.drop(['SJD'],1)
m_tr=m_tr.drop(['SJA'],1)
m_tr=m_tr.drop(['VCH'],1)
m_tr=m_tr.drop(['VCD'],1)
m_tr=m_tr.drop(['VCA'],1)
m_tr=m_tr.drop(['GBD'],1)
m_tr=m_tr.drop(['GBA'],1)
m_tr=m_tr.drop(['BSH'],1)

m_tr=m_tr.drop(['GBH'],1)
m_tr=m_tr.drop(['PSH'],1)
m_tr=m_tr.drop(['PSD'],1)
m_tr=m_tr.drop(['WHD'],1)
m_tr=m_tr.drop(['LBD'],1)
m_tr=m_tr.drop(['LBA'],1)
m_tr=m_tr.drop(['BWH'],1)
m_tr=m_tr.drop(['BWD'],1)
m_tr=m_tr.drop(['BWA'],1)
m_tr=m_tr.drop(['IWH'],1)
m_tr=m_tr.drop(['IWD'],1)
m_tr=m_tr.drop(['IWA'],1)
m_tr=m_tr.drop(['LBH'],1)
m_tr=m_tr.drop(['goal'],1)
m_t=m_test.drop(['date'],1)
m_t=m_t.drop(['BSD'],1)
m_t=m_t.drop(['BSA'],1)
m_t=m_t.drop(['shoton'],1)
m_t=m_t.drop(['shotoff'],1)
m_t=m_t.drop(['foulcommit'],1)
m_t=m_t.drop(['card'],1)
m_t=m_t.drop(['cross'],1)
m_t=m_t.drop(['corner'],1)
m_t=m_t.drop(['possession'],1)
m_t=m_t.drop(['B365H'],1)
m_t=m_t.drop(['B365D'],1)
m_t=m_t.drop(['B365A'],1)
m_t=m_t.drop(['PSA'],1)
m_t=m_t.drop(['WHH'],1)
m_t=m_t.drop(['WHA'],1)
m_t=m_t.drop(['SJH'],1)
m_t=m_t.drop(['SJD'],1)
m_t=m_t.drop(['SJA'],1)
m_t=m_t.drop(['VCH'],1)
m_t=m_t.drop(['VCD'],1)
m_t=m_t.drop(['VCA'],1)
m_t=m_t.drop(['GBD'],1)
m_t=m_t.drop(['GBA'],1)
m_t=m_t.drop(['BSH'],1)

m_t=m_t.drop(['GBH'],1)
m_t=m_t.drop(['PSH'],1)
m_t=m_t.drop(['PSD'],1)
m_t=m_t.drop(['WHD'],1)
m_t=m_t.drop(['LBD'],1)
m_t=m_t.drop(['LBA'],1)
m_t=m_t.drop(['BWH'],1)
m_t=m_t.drop(['BWD'],1)
m_t=m_t.drop(['BWA'],1)
m_t=m_t.drop(['IWH'],1)
m_t=m_t.drop(['IWD'],1)
m_t=m_t.drop(['IWA'],1)
m_t=m_t.drop(['LBH'],1)
m_t=m_t.drop(['goal'],1)
print(m_t.columns)




m=LogisticRegression()
m.fit(m_tr,l_train)
score=m.score(m_t,l_test)
print(score)

l=RandomForestClassifier()
l.fit(m_tr,l_train)
score=l.score(m_t,l_test)
print(score)
score1=l.score(m_tr,l_train)
print(score1)
d=DecisionTreeClassifier()
d.fit(m_tr,l_train)
score1=d.score(m_tr,l_train)
score=d.score(m_t,l_test)
print(score1)
print(score)