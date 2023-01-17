from pandas.api.types import CategoricalDtype

from plotnine import *

from plotnine.data import mpg

%matplotlib inline
# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker
# Reading the file 



path = '../input/pro-kabaddi-2019/'
# Reading the Team_Stat file on which Analysis needs to be done



file = path + 'Team_Stats_Season 7_30th morning.xlsx'

df_Bengal = pd.read_excel(file,'BW')



df_Bengal.head()
df_Bengaluru = pd.read_excel(file,'BB')



df_Bengaluru.head(2)
df_Delhi = pd.read_excel(file,'DD')



df_Delhi.head()
df_Gujrat = pd.read_excel(file,'GF')



df_Gujrat.head()
df_Haryana = pd.read_excel(file,'HS')



df_Haryana.head()
df_Jaipur = pd.read_excel(file,'JPP')



df_Jaipur.head()
df_Patna = pd.read_excel(file,'PPIRATE')



df_Patna.head()
df_Pune = pd.read_excel(file,'PPUL')



df_Pune.head()
df_Tamil = pd.read_excel(file,'TTHA')



df_Tamil.head()
df_Telgu = pd.read_excel(file,'TTITANS')



df_Telgu.head()
df_Mumbai = pd.read_excel(file,'UMUM')



df_Mumbai.head()
df_UP = pd.read_excel(file,'UPYO')



df_UP.head()
df_teams = df_Bengal.append([df_Bengaluru,df_Delhi,df_Gujrat,df_Haryana,df_Jaipur,df_Patna,df_Pune,df_Tamil,df_Telgu,df_Mumbai,df_UP])

df_teams = df_teams.set_index('SEASONS',drop=True)

df_teams = df_teams.reset_index()

df_teams.head(10)
df_teams['WINS%'] = df_teams['WINS']/df_teams['MATCHES PLAYED']

df_teams['DRAWS%'] = df_teams['DRAWS']/df_teams['MATCHES PLAYED']

df_teams['LOSSES%'] = df_teams['LOSSES']/df_teams['MATCHES PLAYED']

df_teams['AVG SUCCESSFUL TACKLES'] = df_teams['SUCCESSFUL TACKLES']/df_teams['MATCHES PLAYED']

df_teams['AVG SUCCESSFUL RAIDS'] = df_teams['SUCCESSFUL RAIDS']/df_teams['MATCHES PLAYED']

df_teams['AVG ALL OUTS INFLICTED'] = df_teams['ALL OUTS INFLICTED']/df_teams['MATCHES PLAYED']

df_teams['AVG ALL OUTS CONCEEDED'] = df_teams['ALL OUTS CONCEEDED']/df_teams['MATCHES PLAYED']
df_teams.head()
df_Bengal_Warrior = df_teams.loc[1:7,:]

df_Bengaluru_Bulls = df_teams.loc[8:15,:]

df_Delhi_Dabangs = df_teams.loc[16:23,:]

df_Gujrat_Fortunegiants = df_teams.loc[24:27,:]

df_Haryana_Steelers = df_teams.loc[28:31,:]

df_Jaipur_Pink_Panthers = df_teams.loc[32:39,:]

df_Patna_Pirates = df_teams.loc[40:47,:]

df_Puneri_Paltan = df_teams.loc[48:55,:]

df_Tamil_Thalaivas = df_teams.loc[56:59,:]

df_Telugu_Titans = df_teams.loc[60:67,:]

df_U_Mumba = df_teams.loc[68:75,:]

df_UP_Yoddha = df_teams.loc[76:,:]
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Bengal_Warrior,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=75)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Bengal_Warrior,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=75)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Bengal_Warrior,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=75)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Bengal_Warrior,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=75)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Bengal_Warrior,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=75)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Bengal_Warrior,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=75)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Bengaluru_Bulls,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Bengaluru_Bulls,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Bengaluru_Bulls,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Bengaluru_Bulls,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Bengaluru_Bulls,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Bengaluru_Bulls,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Delhi_Dabangs,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Delhi_Dabangs,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Delhi_Dabangs,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Delhi_Dabangs,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Delhi_Dabangs,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Delhi_Dabangs,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Gujrat_Fortunegiants,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Gujrat_Fortunegiants,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Gujrat_Fortunegiants,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Gujrat_Fortunegiants,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Gujrat_Fortunegiants,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Gujrat_Fortunegiants,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Haryana_Steelers,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Haryana_Steelers,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Haryana_Steelers,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Haryana_Steelers,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Haryana_Steelers,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Haryana_Steelers,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Jaipur_Pink_Panthers,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Jaipur_Pink_Panthers,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Jaipur_Pink_Panthers,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Jaipur_Pink_Panthers,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Jaipur_Pink_Panthers,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Jaipur_Pink_Panthers,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Patna_Pirates,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Patna_Pirates,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Patna_Pirates,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Patna_Pirates,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Patna_Pirates,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Patna_Pirates,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Puneri_Paltan,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Puneri_Paltan,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Puneri_Paltan,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Puneri_Paltan,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Puneri_Paltan,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Puneri_Paltan,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Tamil_Thalaivas,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Tamil_Thalaivas,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Tamil_Thalaivas,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Tamil_Thalaivas,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Tamil_Thalaivas,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Tamil_Thalaivas,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_Telugu_Titans,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_Telugu_Titans,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_Telugu_Titans,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_Telugu_Titans,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_Telugu_Titans,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_Telugu_Titans,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_U_Mumba,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_U_Mumba,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_U_Mumba,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_U_Mumba,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_U_Mumba,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_U_Mumba,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
fig, axes = plt.subplots(2,3, figsize=(20,15))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'SEASONS', y = 'WINS%', data = df_UP_Yoddha,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'SUCCESS RAID %', data = df_UP_Yoddha,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'SUCCESSFUL TACKLE %', data = df_UP_Yoddha,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL RAIDS', data = df_UP_Yoddha,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'SEASONS', y = 'AVG SUCCESSFUL TACKLES', data = df_UP_Yoddha,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'SEASONS', y = 'AVG ALL OUTS INFLICTED', data = df_UP_Yoddha,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
df_teams_final = df_teams.iloc[[0,8,16,24,28,32,40,48,56,60,68,76],:]

df_teams_final = df_teams_final.rename(columns = {"SEASONS": "Team"})

df_teams_final.head()
fig, axes = plt.subplots(2,3, figsize=(25,16))

fig.tight_layout()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.55)



a=sns.barplot(x = 'Team', y = 'WINS%', data = df_teams_final,ax=axes[0][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'Team', y = 'SUCCESS RAID %', data = df_teams_final,ax=axes[0][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'Team', y = 'SUCCESSFUL TACKLE %', data = df_teams_final,ax=axes[0][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)

a=sns.barplot(x = 'Team', y = 'AVG SUCCESSFUL RAIDS', data = df_teams_final,ax=axes[1][0]) 

b=plt.setp(a.get_xticklabels(), rotation=45)

a1=sns.barplot(x = 'Team', y = 'AVG SUCCESSFUL TACKLES', data = df_teams_final,ax=axes[1][1])

b1=plt.setp(a1.get_xticklabels(), rotation=45)

a2=sns.barplot(x = 'Team', y = 'AVG ALL OUTS INFLICTED', data = df_teams_final,ax=axes[1][2])

b2=plt.setp(a2.get_xticklabels(), rotation=45)
#Correlation of df_teams_final dataframe



plt.rcParams['figure.figsize'] = [20,15]

df_corr = df_teams_final.corr()

ax = sns.heatmap(df_corr,annot=True)



plt.setp(ax.get_xticklabels(), rotation=50)

plt.setp(ax.get_yticklabels(), rotation=50)

ax.title.set_text('Correlation Matrix Key Variables')
(ggplot(df_teams_final)

 + aes(x='WINS%', y='SUCCESSFUL TACKLE %',color='Team' )

 + geom_point()

 + labs(title='Wins% vs. SUCCESSFUL TACKLE %', x='WINS%', y='SUCCESSFUL TACKLE %')

)
(ggplot(df_teams_final)

 + aes(x='WINS%', y='SUCCESS RAID %',color='Team' )

 + geom_point()

 + labs(title='Wins% vs. SUCCESSFUL RAID %', x='WINS%', y='SUCCESSFUL RAID %')

)
(ggplot(df_teams_final)

 + aes(x='WINS%', y='AVG SUCCESSFUL RAIDS',color='Team' )

 + geom_point()

 + labs(title='Wins% vs.AVG SUCCESSFUL RAIDS', x='WINS%', y='AVG SUCCESSFUL RAIDS')

)
(ggplot(df_teams_final)

 + aes(x='WINS%', y='AVG SUCCESSFUL TACKLES',color='Team' )

 + geom_point()

 + labs(title='Wins% vs. AVG SUCCESSFUL TACKLES', x='WINS%', y='AVG SUCCESSFUL TACKLES')

)
(ggplot(df_teams_final)

 + aes(x='WINS%', y='AVG ALL OUTS INFLICTED',color='Team' )

 + geom_point()

 + labs(title='Wins% vs. ALL OUTS INFLICTED', x='WINS%', y='ALL OUTS INFLICTED')

)
df_teams_s7 = df_teams.iloc[[1,9,17,25,29,33,41,49,57,61,69,77],:]

df_teams_s7 = df_teams_s7.rename(columns = {"SEASONS": "Team"})

df_teams_S7 = df_teams_s7

df_teams_S7
plt.rcParams['figure.figsize'] = [12,10]

df_teams_s7 = df_teams_s7.sort_values(by='AVG SUCCESSFUL RAIDS',ascending=False)

a=sns.barplot(x = 'Team', y = 'AVG SUCCESSFUL RAIDS', data = df_teams_s7) 

b=plt.setp(a.get_xticklabels(), rotation=75)
plt.rcParams['figure.figsize'] = [12,10]

df_teams_s7 = df_teams_s7.sort_values(by='AVG SUCCESSFUL TACKLES',ascending=False)

a=sns.barplot(x = 'Team', y = 'AVG SUCCESSFUL TACKLES', data = df_teams_s7) 

b=plt.setp(a.get_xticklabels(), rotation=75)
plt.rcParams['figure.figsize'] = [12,10]

df_teams_s7 = df_teams_s7.sort_values(by='SUCCESSFUL RAIDS',ascending=False)

a=sns.barplot(x = 'Team', y = 'SUCCESSFUL RAIDS', data = df_teams_s7) 

b=plt.setp(a.get_xticklabels(), rotation=75)
plt.rcParams['figure.figsize'] = [12,10]

df_teams_s7 = df_teams_s7.sort_values(by='SUCCESSFUL TACKLES',ascending=False)

a=sns.barplot(x = 'Team', y = 'SUCCESSFUL TACKLES', data = df_teams_s7) 

b=plt.setp(a.get_xticklabels(), rotation=75)
df_teams_raids = df_teams_S7[['Team','MATCHES PLAYED','AVG SUCCESSFUL RAIDS','SUCCESSFUL RAIDS']]

df_teams_raids = df_teams_raids.reset_index(drop=True)

df_teams_raids
# Add column with Name Matches to be played

df_teams_raids['TO BE PLAYED in Group'] = [2,3, 2, 2, 3, 3, 3, 2, 3, 4, 3, 4]



df_teams_raids['TO BE PLAYED in Playoffs'] = [2,1, 2, 0, 2, 0, 0, 0, 0, 0, 1, 2]

df_teams_raids['MATCHES PLAYED'] = df_teams_raids['MATCHES PLAYED'] + df_teams_raids['TO BE PLAYED in Group'] + df_teams_raids['TO BE PLAYED in Playoffs']



df_teams_raids['SUCCESSFUL RAIDS'] = df_teams_raids['AVG SUCCESSFUL RAIDS'] * df_teams_raids['MATCHES PLAYED']
df_teams_raids
plt.rcParams['figure.figsize'] = [12,10]

df_teams_raids = df_teams_raids.sort_values(by='SUCCESSFUL RAIDS',ascending=False)

a=sns.barplot(x = 'Team', y = 'SUCCESSFUL RAIDS', data = df_teams_raids) 

b=plt.setp(a.get_xticklabels(), rotation=75)
df_teams_tackles = df_teams_S7[['Team','MATCHES PLAYED','AVG SUCCESSFUL TACKLES','SUCCESSFUL TACKLES']]

df_teams_tackles = df_teams_tackles.reset_index(drop=True)

df_teams_tackles
# Add column with Name Matches to be played

df_teams_tackles['TO BE PLAYED in Group'] = [2,3, 2, 2, 3, 3, 3, 2, 3, 4, 3, 4]



df_teams_tackles['TO BE PLAYED in Playoffs'] = [2,1, 2, 0, 2, 0, 0, 0, 0, 0, 1, 2]

df_teams_tackles['MATCHES PLAYED'] = df_teams_tackles['MATCHES PLAYED'] + df_teams_tackles['TO BE PLAYED in Group'] + df_teams_tackles['TO BE PLAYED in Playoffs']



df_teams_tackles['SUCCESSFUL TACKLES'] = df_teams_tackles['AVG SUCCESSFUL TACKLES'] * df_teams_tackles['MATCHES PLAYED']

df_teams_tackles
plt.rcParams['figure.figsize'] = [12,10]

df_teams_tackles = df_teams_tackles.sort_values(by='SUCCESSFUL TACKLES',ascending=False)

a=sns.barplot(x = 'Team', y = 'SUCCESSFUL TACKLES', data = df_teams_tackles) 

b=plt.setp(a.get_xticklabels(), rotation=75)
df_teams_s7['SuperPerformance'] =  df_teams_s7['NO. OF SUPER RAIDS'] + df_teams_s7['NO. OF SUPER TACKLES'] + df_teams_s7['ALL OUTS INFLICTED'] - df_teams_s7['ALL OUTS CONCEEDED']
plt.rcParams['figure.figsize'] = [12,10]

df_teams_s7 = df_teams_s7.sort_values(by='SuperPerformance',ascending=False)

a=sns.barplot(x = 'Team', y = 'SuperPerformance', data = df_teams_s7) 

b=plt.setp(a.get_xticklabels(), rotation=75)