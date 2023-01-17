import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import xmltodict
import pprint
import json
import pandas as pd
import os
file='../input/racedata/DMR20180718TCH.xml'
with open(file, 'r') as f:
    xmlString = f.read()
jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)

## The output is quite large.  Commenting it out to surpress for aesthetics.
# print(jsonString)
def extractData(inp):
    data=json.loads(jsonString)
    data=data['CHART']['RACE']
    horsieData=[]
    for race in data:
        run={}
        run['distance']=float(race['DISTANCE'])
        run['purse']=float(race['PURSE'])
        if(race['COURSE_ID']=='D'):
            run['track']=1
        else:
            run['track']=0
        run['fr1']=float(race['FRACTION_1'])
        run['fr2']=float(race['FRACTION_2'])
        run['fr3']=float(race['FRACTION_3'])
        run['fr4']=float(race['FRACTION_4'])
        run['fr5']=float(race['FRACTION_5'])
        run['frw']=float(race['WIN_TIME'])
        horse={}
        for entry in race['ENTRY']:
            horse['pp_start']=int(entry['START_POSITION'])
            horse['pp_finish']=int(entry['OFFICIAL_FIN'])
            if(int(entry['OFFICIAL_FIN'])<4):
                horse['winloss']=1
            else:
                horse['winloss']=0
            if(int(entry['OFFICIAL_FIN'])==1):
                horse['first']=1
                horse['second']=0
                horse['third']=0
                horse['loser']=0
            elif(int(entry['OFFICIAL_FIN'])==2):
                horse['first']=0
                horse['second']=1
                horse['third']=0
                horse['loser']=0
            elif(int(entry['OFFICIAL_FIN'])==3):
                horse['first']=0
                horse['second']=0
                horse['third']=1
                horse['loser']=0
            else:
                horse['first']=0
                horse['second']=0
                horse['third']=0
                horse['loser']=1
            horse['age']=int(entry['AGE'])
            horse['speed_rating']=int(entry['SPEED_RATING'])
            horse['horse']=str(entry['AXCISKEY'])
            horse['jockey name']=str(entry['JOCKEY']['FIRST_NAME'])+' '+str(entry['JOCKEY']['LAST_NAME'])
            horse['trainer name']=str(entry['TRAINER']['FIRST_NAME'])+' '+str(entry['TRAINER']['LAST_NAME'])
            horse['odds']=float(entry['DOLLAR_ODDS'])
            call={}
            for poc in entry['POINT_OF_CALL']:
                if(poc['@WHICH']=='1'):
                    call['poc1']=float(poc['LENGTHS'])
                elif(poc['@WHICH']=='2'):
                    call['poc2']=float(poc['LENGTHS'])
                elif(poc['@WHICH']=='3'):
                    call['poc3']=float(poc['LENGTHS'])
                elif(poc['@WHICH']=='4'):
                    call['poc4']=float(poc['LENGTHS'])
                elif(poc['@WHICH']=='5'):
                    call['poc5']=float(poc['LENGTHS'])
                elif(poc['@WHICH']=='FINAL'):
                    call['pocf']=float(poc['LENGTHS'])
            outp={**run, **horse, **call}
            horsieData.append(outp)
    return horsieData
horsieData=[]

for x in os.listdir('../input/racedata'):
    file='../input/racedata/'+str(x)
    with open(file, 'r') as f:
        xmlString = f.read()
    jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)
    horsieData+=extractData(jsonString)
    
df=pd.DataFrame(horsieData)
df.head()
# df.to_csv('resources/horseData.csv', index=False)
# df=pd.read_csv('allRaces/horseData.csv')
# df.head()
horseLength=8
furlong=660
dist1=1320
dist2=2640
dist3=3960
df['speed1']=((dist1-(horseLength*df['poc1'].astype(float)))/df['fr1'].astype(float)).round(2)
df['speed2']=((dist2-(horseLength*df['poc2'].astype(float)))/df['fr2'].astype(float)).round(2)
df['speed3']=((dist3-(horseLength*df['poc3'].astype(float)))/df['fr3'].astype(float)).round(2)
df['speedf']=(((df['distance'].astype(float)/100*furlong)-(horseLength*df['pocf'].astype(float)))/df['frw'].astype(float)).round(2)
df.head()
data=pd.DataFrame(df[['distance','pp_start','speed1','speed2','speed3','speedf','pp_finish']])
data.head()

data=pd.DataFrame(df[['distance','purse','track','age','jockey name','trainer name','odds','pp_start','winloss']])
data.head()
jockey=pd.read_csv('../input/jockies/jockeyStats.csv')
jockey=jockey.rename(columns={'jockey_name':'jockey name'})
# jockey.sort_values(by='jockey name')

trainer=pd.read_csv('../input/trainers/trainerstats.csv')
trainer=trainer.rename(columns={'trainer_name':'trainer name'})
# trainer.sort_values(by='trainer name')

hj=pd.merge(data,jockey, how='left', left_on='jockey name', right_on='jockey name')
# hj.sort_values(by='jockey name').head()

hjt=pd.merge(hj,trainer, how='left', left_on='trainer name', right_on='trainer name')
hjt.sort_values(by='trainer name').head()
# list(hjt.columns.values)
cols=['distance',
 'purse',
 'track',
 'age',
 'odds',
 'pp_start',
 'j_starts',
 'j_first',
 'j_second',
 'j_third',
 'j_total_win_amt',
 'j_win_perc',
 'j_top_3',
 'j_top_3_perc',
 't_starts',
 't_first',
 't_second',
 't_third',
 't_total_win_amt',
 't_win_perc',
 't_top_3',
 'winloss']
outp_df=hjt[cols]
outp_df.head()

## Surpressing output for kaggle
# outp_df.to_csv('output/horseData_20180804v1.1.csv', index=False)