# *Import libraries*

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as mat

# *Import FIFA 2019 and FIFA 20 dataset. We need to identify the shared columns between the database to make sure that they exactly match*

df= pd.read_csv('../input/fifa19/data.csv')

fifa20 = pd.read_csv("../input/fifa-20-complete-player-dataset/fifa20_data.csv")
fifa20.columns   
df.columns
fifa20.head()
df.head()
df=df.drop(df[['ID','Unnamed: 0','Value','Height','Weight','Wage','Weak Foot','Special','Preferred Foot','Skill Moves','Work Rate','Body Type','Photo','Nationality','Flag','Club Logo','Real Face','Jersey Number','Joined','Loaned From','Contract Valid Until','Release Clause']],axis=1)
difcol20 = fifa20.columns.difference(df.columns)

difcol19 = df.columns.difference(fifa20.columns)
difcol20
difcol19
fifa20.rename(columns={'Ball Control': 'BallControl', 'FK Accuracy': 'FKAccuracy','GK Diving':'GKDiving','GK Handling':'GKHandling','GK Positioning':'GKPositioning','GK Reflexes':'GKReflexes','Heading Accuracy':'HeadingAccuracy','Short Passing':'ShortPassing','Shot Power':'ShotPower','Sliding Tackle':'SlidingTackle','Sprint Speed':'SprintSpeed','Standing Tackle':'StandingTackle','Long Passing':'LongPassing','Long Shots':'LongShots'}, inplace=True)

difcol20 = fifa20.columns.difference(df.columns)

fifa20=fifa20.drop(fifa20[difcol20],axis=1)

difcol19 = df.columns.difference(fifa20.columns)

df=df.drop(df[difcol19],axis=1)
fifa20.columns
df.columns
difcol20 = fifa20.columns.difference(df.columns)

difcol20
difcol19 = df.columns.difference(fifa20.columns)

difcol19
df['New Position']=0

df['New Position'][df['Position']=='GK']='GK'

fifa20['New Position']=0

fifa20['New Position'][fifa20['Position']=='GK']='GK'
gk19 = df[df['New Position']=='GK']

gk20 = fifa20[fifa20['New Position']=='GK']
gk20.head()
gk19.head()
gk19 = gk19.drop(['Name','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',

              'LongPassing','BallControl','Acceleration','SprintSpeed', 'Agility', 'Balance', 'ShotPower',

              'LongShots','Interceptions','Positioning','Vision','Penalties','Marking','StandingTackle','SlidingTackle',

             'Aggression','Stamina'],

            axis=1)



gk20 = gk20.drop(['Name','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',

              'LongPassing','BallControl','Acceleration','SprintSpeed', 'Agility', 'Balance', 'ShotPower',

              'LongShots','Interceptions','Positioning','Vision','Penalties','Marking','StandingTackle','SlidingTackle',

             'Aggression','Stamina'],

            axis=1)
gk19.columns = [str(col) + " GK" for col in gk19.columns]

gk19['Club']=gk19['Club GK']

gk19=gk19.drop(['Club GK'],axis=1)

gk20.columns = [str(col) + " GK" for col in gk20.columns]

gk20['Club']=gk20['Club GK']

gk20=gk20.drop(['Club GK'],axis=1)
gk19.head()
gk20.head()
gk19teams = gk19.groupby('Club').mean().sort_values('Overall GK',ascending=False)

gk19teams.head()
gk20teams = gk20.groupby('Club').mean().sort_values('Overall GK',ascending=False)

gk20teams.head()
notgk19 = df[df['New Position']!='GK']

notgk20 = fifa20[fifa20['New Position']!='GK']
notgk19.columns
notgk19 = notgk19.drop(['Name','GKDiving','GKHandling','GKPositioning','GKReflexes'],axis=1)

notgk20 = notgk20.drop(['Name','GKDiving','GKHandling','GKPositioning','GKReflexes'],axis=1)
notgk19.head()
notgk19teams = notgk19.groupby('Club').mean().sort_values('Overall',ascending=False)

notgk20teams = notgk20.groupby('Club').mean().sort_values('Overall',ascending=False)

notgk19teams
teams19=pd.merge(notgk19teams,gk19teams,'right','Club')

teams20=pd.merge(notgk20teams,gk20teams,'right','Club')

teams19 = teams19.drop(["Potential GK","Jumping GK","GKHandling GK","GKPositioning GK","Reactions GK","Composure GK","GKDiving GK","Volleys","Curve","FKAccuracy","Jumping","LongShots","Penalties",],axis=1)

teams20 = teams20.drop(["Potential GK","Jumping GK","GKHandling GK","GKPositioning GK","Reactions GK","Composure GK","GKDiving GK","Volleys","Curve","FKAccuracy","Jumping","LongShots","Penalties",],axis=1)
uk = pd.read_csv('../input/europe-top-leagues-1819-results/UK.csv',sep=';',encoding='latin-1')

es = pd.read_csv('../input/europe-top-leagues-1819-results/ES.csv',sep=';',encoding='latin-1')

it = pd.read_csv('../input/europe-top-leagues-1819-results/IT.csv',sep=';',encoding='latin-1')

de = pd.read_csv('../input/europe-top-leagues-1819-results/DE.csv',sep=';',encoding='latin-1')

be = pd.read_csv('../input/europe-top-leagues-1819-results/BE.csv',sep=';',encoding='latin-1')

fr = pd.read_csv('../input/europe-top-leagues-1819-results/FR.csv',sep=';',encoding='latin-1')

ne = pd.read_csv('../input/europe-top-leagues-1819-results/NE.csv',sep=';',encoding='latin-1')

pt = pd.read_csv('../input/europe-top-leagues-1819-results/PO.csv',sep=';',encoding='latin-1')

tr = pd.read_csv('../input/europe-top-leagues-1819-results/TR.csv',sep=';',encoding='latin-1')
uk.head()
es.head()
tr.head()
allres = uk.append([be,de,tr,es,ne,fr,pt,it])

allres['Div'].unique()
allres=allres[['HomeTeam','AwayTeam','FTHG','FTAG','FTR']]

allres.head()
allres['HomeTeam'].describe()
HomeStats = teams19

HomeStats = HomeStats.add_prefix('Home ')

HomeStats = HomeStats.reset_index()

AwayStats = teams19

AwayStats = AwayStats.add_prefix('Away ')

AwayStats = AwayStats.reset_index()
HomeStats.head()
AwayStats.head()
res1 = pd.merge(allres,HomeStats,'left',left_on='HomeTeam',right_on='Club')

res1.head()

alltable = pd.merge(res1, AwayStats, 'left',left_on='AwayTeam',right_on='Club')
nan = alltable[alltable['Club_x'].isna()]

nan['HomeTeam'].unique()
allres['HomeTeam'] = allres['HomeTeam'].replace('FC Schalke 04 04', 'FC Schalke 04')

allres['AwayTeam'] =  allres['AwayTeam'].replace('FC Schalke 04 04', 'FC Schalke 04')

allres['HomeTeam'] =  allres['HomeTeam'].replace('Medipol Baþakþehir FK', 'Medipol Başakşehir FK')

allres['AwayTeam'] =  allres['AwayTeam'].replace('Medipol Baþakþehir FK', 'Medipol Başakşehir FK')

allres['HomeTeam'] = allres['HomeTeam'].replace('Beþiktaþ JK', 'Beşiktaş JK')

allres['AwayTeam'] = allres['AwayTeam'].replace('Beþiktaþ JK', 'Beşiktaş JK')

allres['HomeTeam'] = allres['HomeTeam'].replace('Sociedad', 'Real Sociedad')

allres['AwayTeam'] = allres['AwayTeam'].replace('Sociedad', 'Real Sociedad')

allres['HomeTeam'] = allres['HomeTeam'].replace('Spal', 'SPAL')

allres['AwayTeam'] = allres['AwayTeam'].replace('Spal', 'SPAL')

allres['HomeTeam'] = allres['HomeTeam'].replace('Kasimpaþa SK', 'Kasimpaşa SK')

allres['AwayTeam'] = allres['AwayTeam'].replace('Kasimpaþa SK', 'Kasimpaşa SK')
res1 = pd.merge(allres,HomeStats,'left',left_on='HomeTeam',right_on='Club')

alltable2 = pd.merge(res1, AwayStats, 'left',left_on='AwayTeam',right_on='Club')
nan2 = alltable2[alltable2['Club_x'].isna()]

nan2['Club_x'].unique()
nan2 = alltable2[alltable2['Club_y'].isna()]

nan2['Club_y'].unique()
alltable2.info()
alltable2.describe()
table = alltable2.drop(columns=['HomeTeam','AwayTeam','Club_x','Club_y'])

table.head()
table['FTR']= table['FTR'].replace(['H','A','D'],[1,2,0])

table.head()
tablek=table.iloc[:,2:]

tablek.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(tablek.iloc[:,1:])

scaled_feat=scaler.transform(tablek.iloc[:,1:])

tablek_feat=pd.DataFrame(scaled_feat,tablek.iloc[:,1:])

X = tablek_feat

y=tablek['FTR']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=8)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
error_rate=[]



for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))



mat.figure(figsize=(10,6))

mat.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

mat.title('Error Rate vs. K Value')

mat.xlabel('K')

mat.ylabel('Error Rate')
from sklearn import metrics

k_range= range(1,50)



scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))



print(scores)



mat.plot(k_range, scores)

mat.xlabel('Value of K for KNN')

mat.ylabel('Testing Accuracy')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
cl = pd.read_excel('../input/champions-league-groups-1920/clgroups1920.xlsx',header=0)

table1=pd.merge(cl, HomeStats,'left', left_on='HomeTeam',right_on='Club')

clmatches=pd.merge(table1, AwayStats,'left',left_on='AwayTeam',right_on='Club')

clmatches
clmatches.info()
clmatches=clmatches.drop(['Club_x','Club_y'],axis=1)

clmatches.info()
scaler.fit(clmatches.iloc[:,3:])

scaled_feat=scaler.transform(clmatches.iloc[:,3:])

tablecl_feat=pd.DataFrame(scaled_feat,clmatches.iloc[:,3:])

Xcl = tablecl_feat

predcl=knn.predict(Xcl)
clmatches['Results']=predcl

clresults=clmatches[['Group ','HomeTeam','AwayTeam','Results']]

clresults['Homepts']=0

clresults['Awaypts']=0

clresults['Homepts'][clresults['Results']==1]=3

clresults['Awaypts'][clresults['Results']==2]=3

clresults['Homepts'][clresults['Results']==0]=1

clresults['Awaypts'][clresults['Results']==0]=1

clresults[clresults['Group ']=='A']
hpts=clresults.groupby(['Group ','HomeTeam']).sum()

hpts=hpts.drop(['Awaypts','Results'],axis=1)

apts=clresults.groupby(['Group ','AwayTeam']).sum()

apts=apts.drop(['Homepts','Results'],axis=1)
hpts.reset_index(inplace=True)

apts.reset_index(inplace=True)

clpred = pd.concat([hpts,apts],axis=1)

clpred['Total Points']=clpred['Homepts']+clpred['Awaypts']

clpred=clpred.drop(columns=['Homepts','Awaypts','AwayTeam'],axis=1)

clpred=clpred.iloc[:,~clpred.columns.duplicated()]

clpred=clpred.groupby(['Group ','HomeTeam']).sum()

clpred.sort_values(['Group ','Total Points'],ascending=False).groupby('Group ').head(4)
clpred.sort_values(['Group ','Total Points'],ascending=False).groupby('Group ').head(2)