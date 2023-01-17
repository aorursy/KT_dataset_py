# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

np.random.seed(42)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from pathlib import Path #getting out of comfort zone

import matplotlib.pyplot as plt

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH_in=Path("/kaggle/input/graduate-admissions")
ver1=Path(PATH_in/"Admission_Predict.csv")

ver2=Path(PATH_in/"Admission_Predict_Ver1.1.csv")
Df_2=pd.read_csv(ver2,index_col='Serial No.')

Df_2.head()
Df_2.tail()
Df_2.describe().T
Df_2.rename(columns={'University Rating':'University_Rating','GRE Score':'GRE_Score',

                     'TOEFL Score':'TOEFL_Score','Chance of Admit':'Chance_of_Admit'},inplace=True) #Chance of admit does not work

Df_2.head()
col=Df_2.columns

col #due to the space after t in chance of admit and R in LOR (30mins)
Df_2.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance_of_Admit'},inplace=True)

Df_2.head()
figsize=(15,9)
fig,ax=plt.subplots(figsize=figsize)

sns.scatterplot(x=Df_2.loc[Df_2.University_Rating==5]['GRE_Score'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("GRE Score of Students vs Chance of Admit in University with Rating 5")

plt.xlabel("GRE Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.scatterplot(x=Df_2.loc[Df_2.University_Rating==1]['GRE_Score'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("GRE Score of Students vs Chance of Admit in University with Rating 1")

plt.xlabel("GRE Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.scatterplot(x=Df_2.loc[Df_2.University_Rating==5]['CGPA'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("GRE Score of Students vs Chance of Admit in University with Rating 5")

plt.xlabel("CGPA",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.scatterplot(x=Df_2.loc[Df_2.University_Rating==1]['CGPA'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("GRE Score of Students vs Chance of Admit in University with Rating 1")

plt.xlabel("CGPA",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.distplot(Df_2['GRE_Score'],ax=ax)

plt.xlabel("GRE Score",fontsize=20)

fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2['GRE_Score'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("GRE Score of Students vs Chance of Admit in any University",fontsize=15)

plt.xlabel("GRE Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2['CGPA'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("CGPA of Students vs Chance of Admit in any University",fontsize=15)

plt.xlabel("CGPA",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2['TOEFL_Score'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("TOEFL Score of Students vs Chance of Admit in any University",fontsize=15)

plt.xlabel("TOEFL Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2['LOR'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("LOR of Students vs Chance of Admit in any University",fontsize=15)

plt.xlabel("LOR Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2['SOP'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("SOP of Students vs Chance of Admit in any University",fontsize=15)

plt.xlabel("SOP Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.swarmplot(x=Df_2['Research'],y=Df_2['Chance_of_Admit'],ax=ax)

plt.title("Research(1=True, 0=False) vs Chance of Admit in any University",fontsize=15)

plt.xlabel("Research",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
sns.heatmap(Df_2.corr(), annot = True, linewidths=.5, cmap= 'YlGnBu')

plt.title('Correlations', fontsize = 20)

plt.gcf().set_size_inches(12, 7)

plt.show()

fig,ax=plt.subplots(figsize=figsize)

sns.swarmplot(x=Df_2['Research'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("Research(1=True, 0=False) vs Chance of Admit in 5 Rating University",fontsize=15)

plt.xlabel("Research",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.swarmplot(x=Df_2['Research'],y=Df_2.loc[Df_2.University_Rating==3]['Chance_of_Admit'],ax=ax)

plt.title("Research(1=True, 0=False) vs Chance of Admit in 3 Rating University",fontsize=15)

plt.xlabel("Research",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.swarmplot(x=Df_2['Research'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("Research(1=True, 0=False) vs Chance of Admit in 1 Rating University",fontsize=15)

plt.xlabel("Research",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==1]['SOP'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("SOP of Students vs Chance of Admit in University where Rating=1",fontsize=15)

plt.xlabel("SOP Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==3]['SOP'],y=Df_2.loc[Df_2.University_Rating==3]['Chance_of_Admit'],ax=ax)

plt.title("SOP of Students vs Chance of Admit in University where Rating=3",fontsize=15)

plt.xlabel("SOP Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==5]['SOP'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("SOP of Students vs Chance of Admit in University where Rating=5",fontsize=15)

plt.xlabel("SOP Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==1]['LOR'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("LOR of Students vs Chance of Admit in University where Rating=1",fontsize=15)

plt.xlabel("LOR Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==3]['LOR'],y=Df_2.loc[Df_2.University_Rating==3]['Chance_of_Admit'],ax=ax)

plt.title("LOR of Students vs Chance of Admit in University where Rating=3",fontsize=15)

plt.xlabel("LOR Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==5]['LOR'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("LOR of Students vs Chance of Admit in University where Rating=5",fontsize=15)

plt.xlabel("LOR Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==1]['CGPA'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("CGPA of Students vs Chance of Admit in University where Rating=1",fontsize=15)

plt.xlabel("CGPA Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==3]['CGPA'],y=Df_2.loc[Df_2.University_Rating==3]['Chance_of_Admit'],ax=ax)

plt.title("CGPA of Students vs Chance of Admit in University where Rating=3",fontsize=15)

plt.xlabel("CGPA Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==5]['CGPA'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("CGPA of Students vs Chance of Admit in University where Rating=5",fontsize=15)

plt.xlabel("CGPA Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==1]['TOEFL_Score'],y=Df_2.loc[Df_2.University_Rating==1]['Chance_of_Admit'],ax=ax)

plt.title("TOEFL Score of Students vs Chance of Admit in University where Rating=1",fontsize=15)

plt.xlabel("TOEFL Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==3]['TOEFL_Score'],y=Df_2.loc[Df_2.University_Rating==3]['Chance_of_Admit'],ax=ax)

plt.title("TOEFL Score of Students vs Chance of Admit in University where Rating=3",fontsize=15)

plt.xlabel("TOEFL Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)
fig,ax=plt.subplots(figsize=figsize)

sns.regplot(x=Df_2.loc[Df_2.University_Rating==5]['TOEFL_Score'],y=Df_2.loc[Df_2.University_Rating==5]['Chance_of_Admit'],ax=ax)

plt.title("TOEFL Score of Students vs Chance of Admit in University where Rating=5",fontsize=15)

plt.xlabel("TOEFL Score",fontsize=20)

plt.ylabel("Chances of Admit",fontsize=20)