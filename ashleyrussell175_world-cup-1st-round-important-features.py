# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from pandas.tools.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/FIFA 2018 Statistics.csv")
print(df.shape)
df.head()
tabla=[]
for i in df['Team'].unique():
    temp=df[df['Team']==i].reset_index(drop=True)
    temp=pd.concat([temp,df[df['Opponent']==i].reset_index(drop=True)], axis=1)
    tabla.append(temp)
df_stat = pd.concat(tabla).reset_index(drop=True)
print(df_stat.shape)
df_stat.head()
# Lets rename the columns adding to each feature "_A" for the team and _D for the opponent
df_stat.columns=['Date','Team','Opponent','Goal Scored_A','Ball Possession %_A', 'Attempts_A','On-Target_A','Off-Target_A','Blocked_A','Corners_A','Offsides_A','Free Kicks_A','Saves_A','Pass Accuracy %_A',
                'Passes_A','Distance Covered (Kms)_A','Fouls Committed_A','Yellow Card_A','Yellow & Red_A','Red_A','Man of the Match_A','1st Goal_A','Round_A', "PSO_A", "Goals in PSO_A","Own_Goals_A","Own Goal Time_A",'Date_D','Team_D','Opponent_D',
                 'Goal Scored_D','Ball Possession %_D','Attempts_D','On-Target_D','Off-Target_D','Blocked_D','Corners_D', 'Offsides_D','Free Kicks_D','Saves_D','Pass Accuracy %_D',
                 'Passes_D','Distance Covered (Kms)_D','Fouls Committed_D','Yellow Card_D','Yellow & Red_D','Red_D','Man of the Match_D','1st Goal_D','Round_D', "PSO_D", "Goals in PSO_D","Own_Goals_D","Own Goal Time_D"]
df_stat = df_stat.drop(['Date_D','Team_D','Opponent_D'], axis=1)
df_stat.shape
#Let's generate some features
df_stat['Dif_Goals']=df_stat['Goal Scored_A']-df_stat['Goal Scored_D']
df_stat['Dif_OnTarget']=df_stat['On-Target_A']-df_stat['On-Target_D']
df_stat['Dif_1stGoal']=df_stat['1st Goal_A']-df_stat['1st Goal_D']
df_stat['Dif_Corners']=df_stat['Corners_A']-df_stat['Corners_D']
df_stat['Perf_Goalkeeper']=df_stat['Saves_A']/(df_stat['On-Target_D']+df_stat['Goal Scored_D'])
df_stat['Precision']=df_stat['On-Target_A']/df_stat['Off-Target_A']
df_stat['Efectiveness']=df_stat['Goal Scored_A']/df_stat['On-Target_A']
df_stat=df_stat.drop(['Round_A', "PSO_A", "Goals in PSO_A","Own_Goals_A","Own Goal Time_A",'Round_D', "PSO_D", "Goals in PSO_D","Own_Goals_D","Own Goal Time_D"], axis=1)
df_stat.shape
df_stat
corrmat = df_stat.corr()
print(corrmat)
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
cg = sns.clustermap(corrmat,figsize=(23,23), linewidths=.5, cmap=cmap, annot=True)
df_stat['Resultado']='Draw'
df_stat['Resultado'][df_stat['Dif_Goals']>0]='Win'
df_stat['Resultado'][df_stat['Dif_Goals']<0]='Lose'
sns.boxplot(y=df_stat['Dif_OnTarget'],x=df_stat['Resultado'], order=['Win','Draw','Lose'], palette='viridis');
sns.boxplot(y=df_stat['Perf_Goalkeeper'],x=df_stat['Resultado'], order=['Win','Draw','Lose'], palette='viridis');
sns.boxplot(y=df_stat['Efectiveness'],x=df_stat['Resultado'], order=['Win','Draw','Lose'], palette='viridis');
sns.boxplot(y=df_stat['Precision'],x=df_stat['Resultado'], order=['Win','Draw','Lose'], palette='viridis');
plt.ylim(0,3);
resumen = df_stat.groupby(by='Team').mean()
resumen.head()
resumen['Team']=resumen.index
columnas=['Saves_A','Pass Accuracy %_A','Ball Possession %_A','Dif_Goals','Dif_OnTarget','Precision', 'Efectiveness','Perf_Goalkeeper']
X = MinMaxScaler().fit_transform(resumen[columnas])
df_res=pd.DataFrame(data=X, columns=columnas, index=resumen.index)
df_res['Team']=df_res.index
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['Uruguay','Portugal'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['France','Argentina'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['Spain','Russia'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['Crotia','Denmark'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['Sweden','Switzerland'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
# Make the plot
plt.figure(figsize=(10,6))
temp=df_res[df_res['Team'].isin(['Colombia','England'])]
parallel_coordinates(temp, 'Team', colormap='Set1', marker='o')
plt.xticks(rotation=90)
plt.ylim(0,1.5);
