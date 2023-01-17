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
df = pd.read_csv("/kaggle/input/ufcdata/data.csv")

#get all R_fighter B_fighter rows 
dustin_r = df['R_fighter'] == "Dustin Poirier"
max_r = df['R_fighter'] == "Max Holloway"

dustin_b = df['B_fighter'] == "Dustin Poirier"
max_b = df['B_fighter'] == "Max Holloway"

mask = max_r | max_b |  dustin_r |  dustin_b

#filter rows that contain R_fighter, B_fighter
df = df.loc[mask]

#generate a new id column
length_of_results = len(df)
index_col = pd.Series(range(0,length_of_results))
df = df.set_index(index_col)

#hash map of fighters: tko/ko value
fighter_ko = {}

for index in range(0,len(df)):
    
    #is current R index max holloway or dustin poirier
    if (df['R_fighter'][index] == 'Max Holloway') or (df['R_fighter'][index] == 'Dustin Poirier'):
        
        #is the fighter currently in the dictionary?
        if df['R_fighter'][index] in fighter_ko:
            
            #if number of current kos > number of previous kos
            if df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index] > fighter_ko[df['R_fighter'][index]]:
                
                #update ko/tko rate
                fighter_ko[df['R_fighter'][index]] =  df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index]
       
        else:
            #update ko/tko rate
            fighter_ko[df['R_fighter'][index]] =  df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index]
            
    #is current B index max holloway or dustin poirier
    if (df['B_fighter'][index] == 'Max Holloway') or (df['B_fighter'][index] == 'Dustin Poirier'):
        
        #is the fighter currently in the dictionary?
        if df['B_fighter'][index] in fighter_ko:
            
            #if number of current kos > number of previous kos
            if df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]> fighter_ko[df['B_fighter'][index]]:
                
                #update ko/tko rate
                fighter_ko[df['B_fighter'][index]] =  df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]
       
        else:
            #update ko/tko rate
            fighter_ko[df['B_fighter'][index]] =  df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]

df_ko = pd.DataFrame(fighter_ko.items(), columns=['Fighter', 'KO/TKO'])
df_ko

