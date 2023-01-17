import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = r'../input/fpl-top-50-managers-1920/Top_50_fpl.xlsx'

data = pd.read_excel(path)

data.head()
data['Cap_Fail'].mean(axis=0)
data['Total_Hits'].mean(axis=0)
data['Def_Pts'].sum(axis=0)
data['Mid_Pts'].sum(axis=0)
data['For_Pts'].sum(axis=0)
formation_df = data[['3_4_3','3_5_2','4_3_3','4_4_2','4_5_1','5_4_1','5_2_3','5_3_2']]

f = formation_df.sum(axis=0).to_frame()

f.rename(columns={0:'Times'},index={'3_4_3':'3-4-3','3_5_2':'3-5-2','4_3_3':'4-3-3','4_4_2':'4-4-2','4_5_1':'4-5-1','5_4_1':'5-4-1','5_2_3':'5-2-3','5_3_2':'5-3-2'})
round(data['343_avg'].sum(axis=0)/data['343_avg'].astype(bool).sum(axis=0),2)
round(data['352_avg'].sum(axis=0)/data['352_avg'].astype(bool).sum(axis=0),2)
round(data['433_avg'].sum(axis=0)/data['433_avg'].astype(bool).sum(axis=0),2)
round(data['442_avg'].sum(axis=0)/data['442_avg'].astype(bool).sum(axis=0),2)
round(data['451_avg'].sum(axis=0)/data['451_avg'].astype(bool).sum(axis=0),2)
round(data['541_avg'].sum(axis=0)/data['541_avg'].astype(bool).sum(axis=0),2)
round(data['523_avg'].sum(axis=0)/data['523_avg'].astype(bool).sum(axis=0),2)
round(data['532_Avg'].sum(axis=0)/data['532_Avg'].astype(bool).sum(axis=0),2)
plt.hist(data['WC_1'],bins=40)

plt.xticks(range(1,21))

plt.yticks(range(11))

plt.title('1st Wildcard')

plt.xlabel('Gameweek')

plt.ylabel('No. of Managers')

plt.show()
plt.hist(data['WC_2'],bins=40)

plt.xticks(range(21,39))

plt.yticks(range(20))

plt.title('2nd Wildcard')

plt.xlabel('Gameweek')

plt.ylabel('No. of Managers')

plt.show()
round(data['Bench Points'].sum(axis=0)/(50*37),2)
round(data['Total_Cap_Pts'].sum(axis=0)/(50*38),2)
round(data['Total_Hits'].sum(axis=0)/(50),2)
round(data['Goals'].sum(axis=0)/(50*38),2)
round(data['Assists'].sum(axis=0)/(50*38),2)
round(data['CS'].sum(axis=0)/(50*38),2)
data['Most_Capped'].value_counts()