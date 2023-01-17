# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
MRSDR_DF=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')

clms_teams=['Season', 'DayNum', 'TeamID', 'Score', 'vsTeamID', 'vxScore', 'Loc','NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3','vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF']

clms_teams_looser=['Season', 'DayNum', 'vsTeamID', 'vxScore','TeamID', 'Score',  'Loc','NumOT', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3','vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF' , 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast', 'TO', 'Stl', 'Blk', 'PF']

len(MRSDR_DF)

DF_SeasonResult=pd.DataFrame(columns=clms_teams)

Looser=pd.DataFrame(columns=clms_teams_looser)



data_len=len(MRSDR_DF)



#winner

DF_SeasonResult=MRSDR_DF.copy()



#Looser

Looser=MRSDR_DF.copy()

Looser=Looser.replace("H","Z")

Looser=Looser.replace("A","H")

Looser=Looser.replace("Z","A")



DF_SeasonResult.columns=clms_teams

Looser.columns=clms_teams_looser



DF_SeasonResult=DF_SeasonResult.append(Looser)



DF_SeasonResult=DF_SeasonResult[clms_teams]
Bool=DF_SeasonResult['Season']==2003

DF_2003=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2004

DF_2004=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2005

DF_2005=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2006

DF_2006=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2007

DF_2007=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2008

DF_2008=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2009

DF_2009=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2010

DF_2010=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2011

DF_2011=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2012

DF_2012=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2013

DF_2013=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2014

DF_2014=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2015

DF_2015=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2016

DF_2016=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2017

DF_2017=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2018

DF_2018=DF_SeasonResult[Bool]



Bool=DF_SeasonResult['Season']==2019

DF_2019=DF_SeasonResult[Bool]

DF_2003=DF_2003.reset_index(drop=True)

DF_2004=DF_2004.reset_index(drop=True)

DF_2005=DF_2005.reset_index(drop=True)

DF_2006=DF_2006.reset_index(drop=True)

DF_2007=DF_2007.reset_index(drop=True)

DF_2008=DF_2008.reset_index(drop=True)

DF_2009=DF_2009.reset_index(drop=True)

DF_2010=DF_2010.reset_index(drop=True)

DF_2011=DF_2011.reset_index(drop=True)

DF_2012=DF_2012.reset_index(drop=True)

DF_2013=DF_2013.reset_index(drop=True)

DF_2014=DF_2014.reset_index(drop=True)

DF_2015=DF_2015.reset_index(drop=True)

DF_2016=DF_2016.reset_index(drop=True)

DF_2017=DF_2017.reset_index(drop=True)

DF_2018=DF_2018.reset_index(drop=True)

DF_2019=DF_2019.reset_index(drop=True)

DF_2003.to_csv("DF_2003.csv",sep=",",index=False)

DF_2004.to_csv("DF_2004.csv",sep=",",index=False)

DF_2005.to_csv("DF_2005.csv",sep=",",index=False)

DF_2006.to_csv("DF_2006.csv",sep=",",index=False)

DF_2007.to_csv("DF_2007.csv",sep=",",index=False)

DF_2008.to_csv("DF_2008.csv",sep=",",index=False)

DF_2009.to_csv("DF_2009.csv",sep=",",index=False)

DF_2010.to_csv("DF_2010.csv",sep=",",index=False)

DF_2011.to_csv("DF_2011.csv",sep=",",index=False)

DF_2012.to_csv("DF_2012.csv",sep=",",index=False)

DF_2013.to_csv("DF_2013.csv",sep=",",index=False)

DF_2014.to_csv("DF_2014.csv",sep=",",index=False)

DF_2015.to_csv("DF_2015.csv",sep=",",index=False)

DF_2016.to_csv("DF_2016.csv",sep=",",index=False)

DF_2017.to_csv("DF_2017.csv",sep=",",index=False)

DF_2018.to_csv("DF_2018.csv",sep=",",index=False)

DF_2019.to_csv("DF_2019.csv",sep=",",index=False)
MTDR_DF=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

clms_teams=['Season', 'DayNum', 'TeamID', 'Score', 'vsTeamID', 'vxScore', 'Loc','NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3','vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF']

clms_teams_looser=['Season', 'DayNum', 'vsTeamID', 'vxScore','TeamID', 'Score',  'Loc','NumOT', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3','vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF' , 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR','Ast', 'TO', 'Stl', 'Blk', 'PF']

len(MTDR_DF)

DF_TourneyResult=pd.DataFrame(columns=clms_teams)

Looser=pd.DataFrame(columns=clms_teams_looser)



data_len=len(MTDR_DF)



#winner

DF_TourneyResult=MTDR_DF.copy()



#Looser

Looser=MTDR_DF.copy()

Looser=Looser.replace("H","Z")

Looser=Looser.replace("A","H")

Looser=Looser.replace("Z","A")



DF_TourneyResult.columns=clms_teams

Looser.columns=clms_teams_looser



DF_TourneyResult=DF_TourneyResult.append(Looser)



DF_TourneyResult=DF_TourneyResult[clms_teams]
DF_TourneyResult.reset_index()
Bool=DF_TourneyResult['Season']==2003

DF_2003TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2004

DF_2004TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2005

DF_2005TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2006

DF_2006TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2007

DF_2007TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2008

DF_2008TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2009

DF_2009TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2010

DF_2010TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2011

DF_2011TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2012

DF_2012TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2013

DF_2013TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2014

DF_2014TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2015

DF_2015TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2016

DF_2016TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2017

DF_2017TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2018

DF_2018TR=DF_TourneyResult[Bool]



Bool=DF_TourneyResult['Season']==2019

DF_2019TR=DF_TourneyResult[Bool]

DF_2003TR=DF_2003TR.reset_index(drop=True)

DF_2004TR=DF_2004TR.reset_index(drop=True)

DF_2005TR=DF_2005TR.reset_index(drop=True)

DF_2006TR=DF_2006TR.reset_index(drop=True)

DF_2007TR=DF_2007TR.reset_index(drop=True)

DF_2008TR=DF_2008TR.reset_index(drop=True)

DF_2009TR=DF_2009TR.reset_index(drop=True)

DF_2010TR=DF_2010TR.reset_index(drop=True)

DF_2011TR=DF_2011TR.reset_index(drop=True)

DF_2012TR=DF_2012TR.reset_index(drop=True)

DF_2013TR=DF_2013TR.reset_index(drop=True)

DF_2014TR=DF_2014TR.reset_index(drop=True)

DF_2015TR=DF_2015TR.reset_index(drop=True)

DF_2016TR=DF_2016TR.reset_index(drop=True)

DF_2017TR=DF_2017TR.reset_index(drop=True)

DF_2018TR=DF_2018TR.reset_index(drop=True)

DF_2019TR=DF_2019TR.reset_index(drop=True)
DF_2003TR.to_csv("DF_2003TR.csv",sep=",",index=False)

DF_2004TR.to_csv("DF_2004TR.csv",sep=",",index=False)

DF_2005TR.to_csv("DF_2005TR.csv",sep=",",index=False)

DF_2006TR.to_csv("DF_2006TR.csv",sep=",",index=False)

DF_2007TR.to_csv("DF_2007TR.csv",sep=",",index=False)

DF_2008TR.to_csv("DF_2008TR.csv",sep=",",index=False)

DF_2009TR.to_csv("DF_2009TR.csv",sep=",",index=False)

DF_2010TR.to_csv("DF_2010TR.csv",sep=",",index=False)

DF_2011TR.to_csv("DF_2011TR.csv",sep=",",index=False)

DF_2012TR.to_csv("DF_2012TR.csv",sep=",",index=False)

DF_2013TR.to_csv("DF_2013TR.csv",sep=",",index=False)

DF_2014TR.to_csv("DF_2014TR.csv",sep=",",index=False)

DF_2015TR.to_csv("DF_2015TR.csv",sep=",",index=False)

DF_2016TR.to_csv("DF_2016TR.csv",sep=",",index=False)

DF_2017TR.to_csv("DF_2017TR.csv",sep=",",index=False)

DF_2018TR.to_csv("DF_2018TR.csv",sep=",",index=False)

DF_2019TR.to_csv("DF_2019TR.csv",sep=",",index=False)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(16,16)) #heatmap size

sns.heatmap(DF_2003.corr(), annot=True, cmap='plasma', linewidths=.5) 
import numpy as np

import pandas as pd

DF=pd.read_csv("DF_2003.csv")

clms=['Season', 'DayNum', 'TeamID', 'Score', 'vsTeamID', 'vsScore', 'Loc',

       'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast',

       'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3', 'vsFTM',

       'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF']

clms_tm=['Season', 'Games','Win','Lose','TeamID', 'Score', 'vsScore', 

       'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast',

       'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3', 'vsFGA3', 'vsFTM',

       'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl', 'vsBlk', 'vsPF']



temp=pd.DataFrame(columns=clms)

tdtemp=pd.DataFrame(np.zeros(34).reshape(1,34))

tdtemp.columns=clms_tm

TeamDATA=pd.DataFrame(columns=clms_tm)



for Year in range(2003,2020):

    filename=("DF_%d.csv" %  Year)

    filenameTR=("DF_%dTR.csv" %  Year)

    DF=pd.read_csv(filename)

    TR=pd.read_csv(filenameTR)

    TeamDATA=pd.DataFrame(columns=clms_tm)

    

    DB=DF.append(TR)

    DB=DB.reset_index()

    Team_list=DB['TeamID'].unique().tolist()

    for num in range(len(Team_list)):

        TeamID=int(Team_list[num])



        temp=DF[DF['TeamID']==Team_list[num]]

        temp=temp.reset_index()

        Season=temp['Season'][0]

        Games=len(temp)

        Win=temp[temp['Score']>temp['vxScore']]['index'].count()

        Lose=temp[temp['Score']<temp['vxScore']]['index'].count()

        TeamID=temp['TeamID'][0]

        Score=temp['Score'].sum()/Games

        vsScore=temp['vxScore'].sum()/Games

        NumOT=temp['NumOT'].sum()/Games

        FGM=temp['FGM'].sum()/Games

        FGA=temp['FGA'].sum()/Games

        FGM3=temp['FGM3'].sum()/Games

        FGA3=temp['FGA3'].sum()/Games

        FTM=temp['FTM'].sum()/Games

        FTA=temp['FTA'].sum()/Games

        OR=temp['OR'].sum()/Games

        DR=temp['DR'].sum()/Games

        Ast=temp['Ast'].sum()/Games

        TO=temp['TO'].sum()/Games

        Stl=temp['Stl'].sum()/Games

        Blk=temp['Blk'].sum()/Games

        PF=temp['PF'].sum()/Games

        vsFGM=temp['vsFGM'].sum()/Games

        vsFGA=temp['vsFGA'].sum()/Games

        vsFGM3=temp['vsFGM3'].sum()/Games

        vsFGA3=temp['vsFGA3'].sum()/Games

        vsFTM=temp['vsFTM'].sum()/Games

        vsFTA=temp['vsFTA'].sum()/Games

        vsOR=temp['vsOR'].sum()/Games

        vsDR=temp['vsDR'].sum()/Games

        vsAst=temp['vsAst'].sum()/Games

        vsTO=temp['vsTO'].sum()/Games

        vsStl=temp['vsStl'].sum()/Games

        vsBlk=temp['vsBlk'].sum()/Games

        vsPF=temp['vsPF'].sum()/Games



        tdtemp['Season'][0]=Season

        tdtemp['Games'][0]=Games

        tdtemp['Win'][0]=Win

        tdtemp['Lose'][0]=Lose

        tdtemp['TeamID'][0]=TeamID

        tdtemp['Score'][0]=Score

        tdtemp['vsScore'][0]=vsScore

        tdtemp['NumOT'][0]=NumOT

        tdtemp['FGM'][0]=FGM

        tdtemp['FGA'][0]=FGA

        tdtemp['FGM3'][0]=FGM3

        tdtemp['FGA3'][0]=FGA3

        tdtemp['FTM'][0]=FTM

        tdtemp['FTA'][0]=FTA

        tdtemp['OR'][0]=OR

        tdtemp['DR'][0]=DR

        tdtemp['Ast'][0]=Ast

        tdtemp['TO'][0]=TO

        tdtemp['Stl'][0]=Stl

        tdtemp['Blk'][0]=Blk

        tdtemp['PF'][0]=PF

        tdtemp['vsFGM'][0]=vsFGM

        tdtemp['vsFGA'][0]=vsFGA

        tdtemp['vsFGM3'][0]=vsFGM3

        tdtemp['vsFGA3'][0]=vsFGA3

        tdtemp['vsFTM'][0]=vsFTM

        tdtemp['vsFTA'][0]=vsFTA

        tdtemp['vsOR'][0]=vsOR

        tdtemp['vsDR'][0]=vsDR

        tdtemp['vsAst'][0]=vsAst

        tdtemp['vsTO'][0]=vsTO

        tdtemp['vsStl'][0]=vsStl

        tdtemp['vsBlk'][0]=vsBlk

        tdtemp['vsPF'][0]=vsPF

        TeamDATA=TeamDATA.append(tdtemp)

    filename3=("TeamData%d.csv"% Year)

    TeamDATA.to_csv(filename3,index=False)


for Year in range(2003,2020):

    filename=("/kaggle/input/0318data4/TeamData%d.csv" % Year)

    Tdata=pd.read_csv(filename)

    Tdata=Tdata.assign(Winpct=0)

    Tdata['Winpct']=Tdata['Win']/(Tdata['Games'])

    filename2=("TeamData%d.csv" % Year)

    Tdata.to_csv(filename2,index=False)

#TeamScore

year=2003

filename=("/kaggle/input/0318data4/TeamData%d.csv" % year)

df=pd.read_csv(filename)

clms=df.columns



TeamScore=pd.DataFrame(columns=clms)



for year in range(2003,2020):

    filename=("/kaggle/input/0318data4/TeamData%d.csv" % year)

    df=pd.read_csv(filename)

   

    TeamScore=TeamScore.append(df)



TeamScore=TeamScore.replace('',np.nan,)

TeamScore=TeamScore.dropna()

TeamScore.reset_index()

TeamScore.to_csv("TeamScore.csv",sep=",",index=False)
Train=pd.DataFrame(columns=['Season','TeamID','Score','vsTeamID','vsScore','LastSeason'])

TeamScore=pd.read_csv("TeamScore.csv")



for Year in range(2004,2020):

    filename=("DF_%d.csv" % Year)

    df=pd.read_csv(filename)

    

    Season=df['Season']

    TeamID=df['TeamID']

    Score=df['Score']

    vsTeamID=df['vsTeamID']

    vsScore=df['vxScore']

    LastSeason=df['Season']-1

    

    DF=pd.DataFrame(np.zeros(len(df)*6).reshape(len(df),6))

    DF.columns=['Season','TeamID','Score','vsTeamID','vsScore','LastSeason']



    DF['Season']=Season

    DF['TeamID']=TeamID

    DF['Score']=Score

    DF['vsTeamID']=vsTeamID

    DF['vsScore']=vsScore

    DF['LastSeason']=LastSeason



    Train=Train.append(DF)



#TR

    filename=("DF_%dTR.csv" % Year)

    df=pd.read_csv(filename)

    

    Season=df['Season']

    TeamID=df['TeamID']

    Score=df['Score']

    vsTeamID=df['vsTeamID']

    vsScore=df['vxScore']

    LastSeason=df['Season']-1

    

    DF=pd.DataFrame(np.zeros(len(df)*6).reshape(len(df),6))

    DF.columns=['Season','TeamID','Score','vsTeamID','vsScore','LastSeason']



    DF['Season']=Season

    DF['TeamID']=TeamID

    DF['Score']=Score

    DF['vsTeamID']=vsTeamID

    DF['vsScore']=vsScore

    DF['LastSeason']=LastSeason



    Train=Train.append(DF)



    

    

Train=Train.reset_index()





Train=Train.assign(result=1)

for num in range(len(Train)):

    if Train['Score'][num] < Train['vsScore'][num]:

        Train['result'][num]=0





Train.to_csv("Train.csv",sep=",",index=False)

Train
trial=pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv")

trial_cp=pd.DataFrame(columns=['Season','TeamID','vsTeamID','LastSeason'])



for data in trial.iterrows():

    ID_string=trial["ID"][data[0]]

    putdata=([ID_string.split("_")])

    put_df=pd.DataFrame(putdata,columns=["Season","TeamID","vsTeamID"])

    trial_cp=trial_cp.append(put_df,ignore_index=True)



trial_cp['Season']=trial_cp['Season'].astype(np.int64)

trial_cp['TeamID']=trial_cp['TeamID'].astype(np.int64)

trial_cp['vsTeamID']=trial_cp['vsTeamID'].astype(np.int64)



trial_cp['LastSeason']=trial_cp['Season']-1



trial_cp.to_csv("testdata.csv",sep=",",index=False)

TS
import pandas as pd



#Train=pd.read_csv("/kaggle/input/0301data1/Train.csv")

#del Train['index']

#TS=pd.read_csv("/kaggle/input/0301data1/TeamScore.csv")

TS=TeamScore



MyTrain=pd.merge(Train,TS,left_on=['LastSeason','TeamID'],right_on=['Season','TeamID'],how="left")

del MyTrain['Season_y']

MyTrain.columns=['Season', 'TeamID','MyScore', 'vsTeamID','MyvsScore', 'LastSeason', 'result', 'Games','Win','Lose',

       'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']

MyTr=MyTrain[['Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']]



rvlTrain=pd.merge(Train,TS,left_on=['LastSeason','vsTeamID'],right_on=['Season','TeamID'],how="left")

del rvlTrain['Season_y']

del rvlTrain['TeamID_y']

rvlTrain.columns=['Season', 'TeamID','MyScore', 'vsTeamID','MyvsScore', 'LastSeason', 'result', 'Games','Win','Lose',

       'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']

rvlTr=rvlTrain[['Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']]

Tr2=MyTr-rvlTr



Tr2.assign(result=0)

Tr2['result']=MyTrain['result']

Tr2.assign(Season=0)

Tr2['Season']=MyTrain['Season']

Tr2.assign(TeamID=0)

Tr2['TeamID']=MyTrain['TeamID']

Tr2.assign(vsTeamID=0)

Tr2['vsTeamID']=MyTrain['vsTeamID']

Tr2.assign(MyWinpct=0)

Tr2['MyWinpct']=MyTrain['Winpct']

Tr2.assign(MyScore=0)

Tr2['MyScore']=MyTrain['Score']

Tr2.assign(MyFGM=0)

Tr2['MyFGM']=MyTrain['FGM']



Tr2=Tr2[[ 'result','Season', 'TeamID','vsTeamID','MyWinpct', 'MyScore','MyFGM', 'Winpct',

             'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF' ]]



#Tr2=Tr2[Tr2['Season']>2009]



Tr2.to_csv("Train_data.csv",sep=",",index=False)
Tr2
trial=pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv")

trial_cp=pd.DataFrame(columns=["Season","TeamID","vsTeamID"])



for data in trial.iterrows():

    ID_string=trial["ID"][data[0]]

    putdata=([ID_string.split("_")])

    put_df=pd.DataFrame(putdata,columns=["Season","TeamID","vsTeamID"])

    trial_cp=trial_cp.append(put_df,ignore_index=True)

#    print(trial_cp)  



trial_cp=trial_cp.assign(LastSeason=0)

for data in trial_cp.iterrows():

    trial_cp['LastSeason'][data[0]]=int(data[1]['Season'])-1

    



    

trial_cp['Season']=trial_cp['Season'].astype(np.int64)

trial_cp['TeamID']=trial_cp['TeamID'].astype(np.int64)

trial_cp['vsTeamID']=trial_cp['vsTeamID'].astype(np.int64)



#TS=pd.read_csv("/kaggle/input/0301data1/TeamScore.csv")





#

MyTest=pd.merge(trial_cp,TS,left_on=['LastSeason','TeamID'],right_on=['Season','TeamID'],how="left")

del MyTest['Season_y']

MyTest.columns=['Season', 'TeamID', 'vsTeamID', 'LastSeason', 'Games','Win','Lose',

       'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']

MyTst=MyTest[['Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']]



rvlTest=pd.merge(trial_cp,TS,left_on=['LastSeason','vsTeamID'],right_on=['Season','TeamID'],how="left")

del rvlTest['Season_y']

del rvlTest['TeamID_y']

rvlTest.columns=['Season', 'TeamID', 'vsTeamID', 'LastSeason', 'Games','Win','Lose',

       'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']

rvlTst=rvlTest[['Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF','Winpct']]





Tst2=MyTst-rvlTst

Tst2.assign(Season=0)

Tst2['Season']=MyTest['Season']

Tst2.assign(TeamID=0)

Tst2['TeamID']=MyTest['TeamID']

Tst2.assign(vsTeamID=0)

Tst2['vsTeamID']=MyTest['vsTeamID']

Tst2.assign(MyWinpct=0)

Tst2['MyWinpct']=MyTest['Winpct']

Tst2.assign(MyScore=0)

Tst2['MyScore']=MyTest['Score']

Tst2.assign(MyFGM=0)

Tst2['MyFGM']=MyTest['FGM']



Tst2=Tst2[[ 'Season', 'TeamID','vsTeamID','MyWinpct', 'MyScore','MyFGM', 'Winpct',

             'Score', 'vsScore', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'vsFGM', 'vsFGA', 'vsFGM3',

       'vsFGA3', 'vsFTM', 'vsFTA', 'vsOR', 'vsDR', 'vsAst', 'vsTO', 'vsStl',

       'vsBlk', 'vsPF' ]]





Tst2.to_csv("Test_data.csv",sep=",",index=False)
import pandas as pd

import numpy as np

#Tr2=pd.read_csv("/kaggle/input/0301data2/Train_data.csv")

Tr2=Tr2.fillna(9999)





y=Tr2['result']

#x=Tr2[['MyScore','MyFGM','MyWinpct','Winpct']]

x=Tr2.copy()

del x['result']

#x=Tr2[['AdvScore', 'vsAdvScore','Advantage']]





from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.2)

x_train.shape,x_test.shape,y_train.shape,y_test.shape









from sklearn.preprocessing import StandardScaler

#sc=StandardScaler()

#x_train_std=sc.fit_transform(x_train)

#x_test_std=sc.transform(x_test)



#x_train.shape,x_test.shape 





import pandas as pd

import numpy as np

#Tst2=pd.read_csv("/kaggle/input/0301data2/Test_data.csv")

Tst2=Tst2.fillna(9999)

#X=Tst2[['MyScore','MyFGM','MyWinpct','Winpct']].values.tolist()

#X=Tst2[['AdvScore', 'vsAdvScore', 'Advantage']]

#X=Tst2.values.tolist()

X=Tst2.values



#X_std=sc.transform(X)



    
Tr2
Tst2
import xgboost as xgb

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

dtrain=xgb.DMatrix(x_train, label=y_train)

dvalid=xgb.DMatrix(x_test, label=y_test)

dtest=xgb.DMatrix(X)



#params={'objective':'binary:logistic','eval_metric': 'logloss','silent':1,'random_state':71}

params={'objective':'binary:logistic','eval_metric': 'rmse','silent':1,'random_state':71}

num_round=20000



watchlist=[(dtrain,'train'),(dvalid,'eval')]

model=xgb.train(params,dtrain,num_round,evals=watchlist,early_stopping_rounds=5000 )



va_pred=model.predict(dvalid)

#score=log_loss(y_test,va_pred)

#print(f'logloss: {score:.4f}')



va_pred_a = np.where(va_pred > 0.5, 1, 0)

# 精度 (Accuracy) を検証する

acc = accuracy_score(y_test, va_pred_a)

print('Accuracy:', acc)





import numpy as np

pred=model.predict(dtest)

output=np.zeros(len(pred)*2).reshape(len(pred),2)

for num in range(len(pred)):

    output[num,0]=num+1

    output[num,1]=pred[num]

    



    

sub=pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv")



sub['Pred']=pred

sub.to_csv("out_xgboost.csv",sep=",",index=False)