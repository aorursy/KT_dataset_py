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
#load data

df = pd.read_csv("/kaggle/input/ufcdata/data.csv")

#store fighter: num_ko/tko

fighter_ko = {}



for index in range(0,len(df)):



    #is the fighter currently in the dictionary?

    if df['R_fighter'][index] in fighter_ko:



        #if number of current kos > number of previous kos

        if df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index] > fighter_ko[df['R_fighter'][index]]:



            #update ko/tko rate

            fighter_ko[df['R_fighter'][index]] =  df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index]



    else:

        #update ko/tko rate

        fighter_ko[df['R_fighter'][index]] =  df['R_win_by_KO/TKO'][index] + df['R_win_by_TKO_Doctor_Stoppage'][index]



    #is the fighter currently in the dictionary?

    if df['B_fighter'][index] in fighter_ko:



        #if number of current kos > number of previous kos

        if df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]> fighter_ko[df['B_fighter'][index]]:



            #update ko/tko rate

            fighter_ko[df['B_fighter'][index]] =  df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]



    else:

        #update ko/tko rate

        fighter_ko[df['B_fighter'][index]] =  df['B_win_by_KO/TKO'][index] + df['B_win_by_TKO_Doctor_Stoppage'][index]

#convert to data frame to sort in descending order

df_ko = pd.DataFrame(fighter_ko.items(), columns=['Fighter', 'KO/TKO'])

df_ko.sort_values(by=['KO/TKO'],ascending=False)