# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/World_Champs_Men\'s_All-Around.csv')
df.head(5)
list_apparatus = df.Apparatus.unique().tolist()

list_apparatus
df_Total_vs_Apparatus = df[['Apparatus', 'Total']].copy()

df_Total_vs_Apparatus.head(10)



df_Total_vs_Apparatus.sort_values('Apparatus')



mean_time_per_apparatus = df_Total_vs_Apparatus.groupby(['Apparatus'])['Total'].mean()

mean_time_per_apparatus
mean_time_per_apparatus.plot(title='Mean Total Time per Apparatus')
df_Difficulty_Score_vs_Apparatus = df[['Apparatus','Diff']].copy()

df_Difficulty_Score_vs_Apparatus.head(5)
max_difficulty_score_per_apparatus = df_Difficulty_Score_vs_Apparatus.groupby(['Apparatus'])['Diff'].max()

max_difficulty_score_per_apparatus
max_difficulty_score_per_apparatus.plot(title='Max Difficulty Score per Apparatus')
df_Name_vs_Rank = df[['Name','Rank']].copy()

df_Name_vs_Rank.head(5)
df_Name_vs_Rank['Rank_copy'] = df_Name_vs_Rank['Rank']

df_Name_vs_Rank.head(5)
max_min_rank_per_name = df_Name_vs_Rank.groupby('Name').agg({'Rank':'min', 'Rank_copy':'max'})[['Rank', 'Rank_copy']].reset_index().rename(columns={'Rank':'Rank_min','Rank_copy':'Rank_max'})

max_min_rank_per_name.head(5)
df_Name_vs_Nationality = df[['Name','Nationality']].copy()

df_Name_vs_Nationality.drop_duplicates()
max_min_rank_per_name.sort_values('Name', ascending=False)[['Rank_min','Rank_max']].plot.bar(stacked=True)