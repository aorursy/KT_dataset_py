# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#
# Hi, i am practising the Data Science Concepts and below is the program working with for loops
# that evaluates to find out the Top "N" states of a day, that have recorded highest number of 
# COVID-19 cases in India from March,14 to May,5 2020 
#
import pandas as pd
import numpy as np

df=pd.read_csv("../input/india-covid19-statewise-data/IndiaStatewiseCovidCases.csv",header=None)

df=df.values
TOP_NO_OF_STATES_TO_FINDOUT=int(input("How Many States having highest Covid Cases of a Day are to be Seen ::"))
TOTAL_NO_OF_DAYS=df.shape[1]

states_final_list=[]
top_states_of_day={}


for day in range(1,TOTAL_NO_OF_DAYS):
    day_list=list(df[1:,(0,day)])
    states_final_list.clear()
    for n in range(0,TOP_NO_OF_STATES_TO_FINDOUT):
        maxval=0
        maxi=0
        for i in range(0,len(day_list)):
            if int(day_list[i][1])>int(maxval):
                maxval=int(day_list[i][1])
                maxi=i
        state_found=day_list.pop(maxi)
        states_final_list.append(state_found)
    top_states_of_day.update({df[0,day]:states_final_list.copy()})
    
for i in top_states_of_day.items():
    idx=0
    print("\nDate:",i[0])
    for j in i[1]:
        idx+=1
        print(idx,j[0],"-->",j[1])