# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import math
import scipy as sp
import scipy.stats# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#import data

data2018 = pd.read_excel('../input/spring2018/2018-spring-match-data-OraclesElixir-2018-02-19.xlsx')
data2017 = pd.read_excel('../input/2017complete/2017matchdataOraclesElixir.xlsx')
print (data2018.columns)
data2017.columns == data2018.columns
data2017.patchno.unique()
data2017 = data2017[data2017['patchno'] > 7.06]
data2017.patchno.unique()
#Create the complete dataset
data = pd.concat([data2017, data2018], ignore_index=True)
data.head()
#Select what is needed
df1 = data[['gameid','gamelength','patchno','champion','result','team','player','csat10','oppcsat10','csdat10','csat15','oppcsat15','csdat15','ft','firstmidouter','fttime','goldat10', 'oppgoldat10', 'gdat10',
       'goldat15', 'oppgoldat15', 'gdat15','position']]
df1.head(12)
dataAzir = df1[df1['champion'] == ('Azir')]
dataAzir.head(12)
dataTal = df1[df1['champion'] == ('Taliyah')]
dataTal.head(12)
dataGalio = df1[df1['champion'] == ('Galio')]
dataGalio.head(12)
dataVS = dataAzir.merge(dataTal, on='gameid', left_index = True, how='inner')
dataVS.head(12)

dataVS = pd.concat([dataAzir,dataTal])
dataVS.head(12)
test = pd.concat([dataAzir,dataTal]).sort_index()

test2 = test['gameid'].value_counts().to_frame()
test2 = test2[test2['gameid'] >1]
test2 = test2.index.values
test2
#Garde que celle dont le compte est >2
dataVS = dataVS[dataVS['gameid'].apply(lambda x: x in test2)]
dataVS['gameid'].apply(lambda x: x in test2  ).unique()

#Garde que celles qui sont vraiment correctes!
dataVS = dataVS[dataVS['gameid'].apply(lambda x: x in (550293,
       560214, 1002450187, 560111, 550946) )]
dataVS
dataVS['gdat10']/dataVS['csdat10']
import matplotlib.pyplot as plt
plt.boxplot(dataAzir['gamelength'] )
dataAzir['gamelength'].mean()
dataAzir['gamelength'].median()
dataAzirEarly = dataAzir[dataAzir['gamelength'] <= dataAzir['gamelength'].median()]
dataAzirEarly['result'].sum()/dataAzirEarly['result'].count()
dataAzirLate = dataAzir[dataAzir['gamelength'] >= dataAzir['gamelength'].median()]
dataAzirLate['result'].sum()/dataAzirLate['result'].count()
dataAzir['result'].sum()/dataAzir['result'].count()
plt.boxplot(dataTal['gamelength'] )
dataTal['gamelength'].mean()

dataTal['gamelength'].median()
dataTalEarly = dataTal[dataTal['gamelength'] <= dataTal['gamelength'].median()]
dataTalEarly['result'].sum()/dataTalEarly['result'].count()
dataTalLate = dataTal[dataTal['gamelength'] >= dataTal['gamelength'].median()]
dataTalLate['result'].sum()/dataTalLate['result'].count()
dataTal['result'].sum()/dataTal['result'].count()
runes = data[data['patchno'] > 8 ]
runes = runes[runes['champion'] == 'Galio' ]
runes = runes[runes['team'] == 'SK Telecom T1' ]
runes
#runes.iloc[11]['url']
#runes['url'] != runes.iloc[11]['url']
data[data['gameid']== 560570] ['team']
dataVS2 = pd.concat([dataGalio,dataTal])
dataVS2 = dataVS2[dataVS2['position'] == 'Middle']
dataVS2.head(12)

test = pd.concat([dataGalio,dataTal]).sort_index()
test = test[test['position'] == 'Middle']
test
test2 = test['gameid'].value_counts().to_frame()
test2 = test2[test2['gameid'] >1]
test2 = test2.index.values
test2
#Garde que celle dont le compte est >2
dataVS2 = dataVS2[dataVS2['gameid'].apply(lambda x: x in test2)]
dataVS2['gameid'].apply(lambda x: x in test2  ).unique()

dataVS2 = dataVS2[dataVS2['gameid'].apply(lambda x: x in (310159, 250330, 1002200135, 1002310239, 300758, 570141, 540826, 360260, 250969,
       260140, 1002230190, 520038, 250740,
       560235, 230055, 550172, 310560, 1002410113, 1002170151, 370285, 230383, 310214, 240148, 530078, 1002280165, 250885, 540099,
       1002110175, 160048, 1002450029, 490057) )]
dataVS2
dataVS2.sort_values('gameid')

dataG = dataVS2[dataVS2['champion'] == 'Galio']
dataG 
dataT = dataVS2[dataVS2['champion'] == 'Taliyah']
dataT
#from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#iris = load_iris()
X, y = dataT[['csat10','gamelength','patchno','csat10','oppcsat10','csdat10','csat15','oppcsat15','csdat15','ft','firstmidouter','fttime','goldat10', 'oppgoldat10', 'gdat10',
       'goldat15', 'oppgoldat15', 'gdat15']], dataT['result']  
X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
X = dataT[['csat10','gamelength','patchno','csat10','oppcsat10','csdat10','csat15','oppcsat15','csdat15','ft','firstmidouter','fttime','goldat10', 'oppgoldat10', 'gdat10',
       'goldat15', 'oppgoldat15', 'gdat15']] 
Y = dataT['result']
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
X = data[['csat10','gamelength','patchno','csat10','oppcsat10','csdat10','csat15','oppcsat15','csdat15','ft','firstmidouter','fttime','goldat10', 'oppgoldat10', 'gdat10',
       'goldat15', 'oppgoldat15', 'gdat15']]
Y = data['result']
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
dataT['gamelength'].median()
plt.boxplot(dataT['gamelength'])
tri = dataT[dataT['gamelength'] > dataT['gamelength'].median() ]
tri['result'].sum()/tri['result'].count()
triTot = dataTal[dataTal['gamelength'] > dataTal['gamelength'].median() ]
triTot['result'].sum()/triTot['result'].count()
triTot2 = dataGalio[dataGalio['gamelength'] > dataGalio['gamelength'].median() ]
triTot2['result'].sum()/triTot2['result'].count()
tri2 = dataT[dataT['firstmidouter'] == 0 ]
tri2['result'].sum()/tri2['result'].count()
tri2Tot = dataTal[dataTal['firstmidouter'] == 1 ]
tri2Tot['result'].sum()/tri2Tot['result'].count()
tri2Tot2 = dataGalio[dataGalio['firstmidouter'] == 1 ]
tri2Tot2['result'].sum()/tri2Tot2['result'].count()
tri3 = dataT[dataT['ft'] == 0 ]
tri3['result'].sum()/tri3['result'].count()
tri3Tot = dataTal[dataTal['ft'] == 1 ]
tri3Tot['result'].sum()/tri3Tot['result'].count()
tri3Tot2 = dataGalio[dataGalio['ft'] == 1 ]
tri3Tot2['result'].sum()/tri3Tot2['result'].count()

tri4 = data[data['firstmidouter'] == 0 ]
tri4['result'].sum()/tri4['result'].count()
tri4 = data[data['ft'] == 1 ]
tri4['result'].sum()/tri4['result'].count()
dataT[dataT['gamelength'] > 45 ]
plt.boxplot(dataT['csdat10'])
dataT['csdat10'].mean()
dataT['csdat10'].median()
plt.boxplot(dataT['gdat10'])
dataT['gdat10'].mean()
potKills = dataT['gdat10'] - dataT['csdat10']*19,74
potKills
potKills.mean()
dataT['ft'].sum()/dataT['ft'].count()
dataTal['ft'].sum()/dataTal['ft'].count()
dataGalio['ft'].sum()/dataGalio['ft'].count()
dataT['result'].sum()/dataT['result'].count()
dataTal['result'].sum()/dataTal['result'].count()
dataGalio['result'].sum()/dataGalio['result'].count()