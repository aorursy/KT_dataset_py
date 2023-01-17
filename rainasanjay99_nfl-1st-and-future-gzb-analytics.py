import numpy as np

from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sb



from pylab import rcParams

rcParams['figure.figsize'] = 50, 10


nfl_pl = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")
nfl_pl['StadiumType']=nfl_pl['StadiumType'].replace(to_replace =['Outdoors', 'Outddors', 'Oudoors','Oudoor', 'Ourdoor', 'Outdor', 'Outside', 'Cloudy', 'Outdoor Retr Roof-Open','Indoors','Indoor', 'Indoor, Open Roof', 'Indoor, Roof Closed'], 

                            value =["Outdoor", "Outdoor", "Outdoor","Outdoor", "Outdoor", "Outdoor", "Outdoor", "Outdoor","Outdoor", "Indoor", "Indoor", "Indoor", "Indoor"])



nfl_pl['StadiumType']=nfl_pl['StadiumType'].replace(to_replace =['Dome', 'Dome, closed', 'Domed', 'Domed, Open', 'Domed, closed', 'Domed, open', 'Closed Dome'], 

                            value =["Dome", "Dome","Dome","Dome","Dome","Dome","Dome"])



nfl_pl['StadiumType']=nfl_pl['StadiumType'].replace(to_replace =['Retractable Roof', 'Retr. Roof - Closed', 'Retr. Roof - Open', 'Retr. Roof Closed', 'Retr. Roof-Closed', 'Retr. Roof-Open'], 

                            value =["Retractable Roof", "Retractable Roof","Retractable Roof","Retractable Roof","Retractable Roof","Retractable Roof",])

del nfl_pl['Position'],nfl_pl['PositionGroup'],nfl_pl['Temperature']

nfl_ir = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

df_m_ir_pl = pd.merge(nfl_pl, nfl_ir, on = ['PlayerKey','GameID','PlayKey'], how='inner')

print (df_m_ir_pl.shape)

print (df_m_ir_pl.columns)

print (df_m_ir_pl.groupby('RosterPosition').size())
nfl_ptd = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")


df_m_ptd_2mrg = pd.merge(df_m_ir_pl, nfl_ptd, on = 'PlayKey')

nfl_final_1 = df_m_ptd_2mrg

print (nfl_final_1.shape)

nfl_final_1 = nfl_final_1.dropna()

print (nfl_final_1.shape)

nfl_final_1.rename(columns = {'s':'Speed', 'o':'Orientation'}) 



print(nfl_final_1.columns)
nfl_final = nfl_final_1

print(nfl_final.columns)



print ('FacetGrid to characterize any differences in player movement between the playing surfaces with - plot for further tweaking')

sb.relplot(x='x',y='y',hue='Surface',size='RosterPosition',col='PlayType',row='event',height=5, data=nfl_final)

print ('FacetGrid identify specific scenarios like weather, position etc. that interact with player movement to present an elevated risk of injury - with plot for further tweaking')

sb.relplot(x='x',y='y',hue='BodyPart',size='Weather',col='RosterPosition',row='PlayType',height=5, data=nfl_final)