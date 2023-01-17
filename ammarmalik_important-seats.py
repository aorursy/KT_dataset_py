import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
## 2002 Elections
Winner02 = pd.DataFrame([])
NA2002 = pd.read_csv("../input/National Assembly 2002 - Updated.csv", encoding = "ISO-8859-1")
Constituencies = NA2002.Constituency_title.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
    NA = NA2002.loc[NA2002['Constituency_title']==Const, ['Constituency_title','Party','Votes', 'Seat']]
    NA = NA.sort_values(by=['Votes'], ascending = False)
#    NA['TotalVotes'] = NA.Votes.sum()
    NA = NA.iloc[0:1,:] # Only Winner
    Winner02 = pd.concat([Winner02,NA])

Winner02 = Winner02.loc[Winner02['Party']=='Pakistan Muslim League(QA)',['Constituency_title', 'Seat']]
Winner02 = Winner02.rename(columns = {'Constituency_title':'ConstituencyTitle', 'Seat':'Seat02'}) #

## 2008 Elections
Winner08 = pd.DataFrame([])
NA2008 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
Constituencies = NA2008.ConstituencyTitle.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
    NA = NA2008.loc[NA2008['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes', 'Seat']]
    NA = NA.sort_values(by=['Votes'], ascending = False)
#    NA['TotalVotes'] = NA.Votes.sum()
    NA = NA.iloc[0:1,:] # Only Winner
    Winner08 = pd.concat([Winner08,NA])

Winner08 = Winner08.loc[Winner08['Party']=='Pakistan Peoples Party Parliamentarians',['ConstituencyTitle', 'Seat']]
Winner08 = Winner08.rename(columns = {'ConstituencyTitle':'ConstituencyTitle', 'Seat':'Seat08'}) #

## 2013 Elections
Winner13 = pd.DataFrame([])
NA2013 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")
Constituencies = NA2013.ConstituencyTitle.unique()
for i in range(len(Constituencies)):
    Const = Constituencies[i]
    NA = NA2013.loc[NA2013['ConstituencyTitle']==Const, ['ConstituencyTitle','Party','Votes', 'Seat']]
    NA = NA.sort_values(by=['Votes'], ascending = False)
#    NA['TotalVotes'] = NA.Votes.sum()
    NA = NA.iloc[0:1,:] # Only Winner
    Winner13 = pd.concat([Winner13,NA])

Winner13 = Winner13.loc[Winner13['Party']=='Pakistan Muslim League (N)',['ConstituencyTitle', 'Seat']]
Winner13 = Winner13.rename(columns = {'ConstituencyTitle':'ConstituencyTitle', 'Seat':'Seat13'}) #

t1 = Winner02
t2 = Winner08
t3 = Winner13
Winner02.reset_index(inplace=True)
Winner02 = Winner02.drop(['index'], axis=1)
Winner08.reset_index(inplace=True)
Winner08 = Winner08.drop(['index'], axis=1)
Winner13.reset_index(inplace=True)
Winner13 = Winner13.drop(['index'], axis=1)
Winner02 = Winner02.set_index('ConstituencyTitle')
Winner08 = Winner08.set_index('ConstituencyTitle')
Winner13 = Winner13.set_index('ConstituencyTitle')
Win_02_08 = pd.merge(Winner02,Winner08, left_index=True, right_index=True)
Win_08_13 = pd.merge(Winner08,Winner13, left_index=True, right_index=True)
Win_02_08_13 = pd.merge(Win_02_08,Winner13, left_index=True, right_index=True)
Win_02_08
Win_08_13
New_Const = pd.DataFrame(['NA-101','NA-102','NA-104', 'NA-100', 'Dissolved', 'NA-72', 'NA-134', 'NA-141', 'NA-144', 'Dissolved', 'NA-154', 'NA-153', 'NA-162', 'NA-185', 'NA-174', 'NA-173', 'NA-168'])
#Win_02_08_13
Win_02_08_13.reset_index(level=0, inplace=True)
Output = pd.concat([Win_02_08_13,New_Const], axis = 1)
Output.rename(columns = {'ConstituencyTitle': 'OldConstituencyTitle',0:'NewConstituencyTitle'}) #