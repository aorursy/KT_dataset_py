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
# that helps in observing the states having Continous increase in the COVID-19 Cases for "N" Days
# Showing the Below Details of that Particular state.
# 1. No. of Continous increased days COVID-19 cases recorded i.e., 'N' or More Days as requested. 
# 2. Highest No. of cases recorded in that state as per the data.  
# 3. No. of Continous decreased days [includes days with zero cases recorded also as it also indicates decrease] 
# 4. No. of maximum continous days observed with zero cases recorded.[helps to compare with Continous Decreased days]  
# 5. Gives the Average of last 7 days of recorded cases.  
#  
#  Worked on COVID-19 cases data of India from March,14 to May,5 2020
import pandas as pd
import numpy as np

df=pd.read_csv("../input/india-covid19-statewise-data/IndiaStatewiseCovidCases.csv",header=None)

df=df.values
### from march 14 th to till data of India's COVID-19 cases
### which states has 'N' days continous increase and after that decrease in number of cases 

NO_OF_DAYS_TO_VERIFY=int(input("how many days of continuous incline to be identified ::"))
NO_OF_LAST_DAYS_FOR_AVG=int(input("how many days of last days average to be calculated ::")) 
TOTAL_STATES_COUNT=df.shape[0]

statewise_details={}
states_list=list(df[1:,0])

# Loop to traverse through States Data
for state in range(1,TOTAL_STATES_COUNT):
    
    current_data=list(df[state,1:])
    current_state=states_list[state-1] 
    
    continous_increase=0
    continous_decrease=0
    continous_zeroes=0
    
    max_increased_days=0
    max_decreased_days=0
    max_continous_zeroes=0
    
    highest_recorded_cases=0
    last_days_avg=0
    
   # Loop for computations on Data of the Current State  
    for i in range(0,len(current_data)):
        
        # Computations for 1st or 0th elements of the list and continue to next element by skipping further steps
        if i == 0:
            if int(current_data[i])==0:
                continous_zeroes=1
            else:
                highest_recorded_cases=int(current_data[i])
            continue
            
        # Computations for finding last N days Average No. of cases    
        if i >= len(current_data)-NO_OF_LAST_DAYS_FOR_AVG:
            last_days_avg+=int(current_data[i])
        if i == len(current_data)-1:
            last_days_avg/=NO_OF_LAST_DAYS_FOR_AVG
        
        # To record Highest No. of Cases observed in the Current State    
        if highest_recorded_cases < int(current_data[i]):
            highest_recorded_cases=int(current_data[i])
        
        # Comparison to identify the incline / decline of the cases
        if int(current_data[i-1])<int(current_data[i]):
            continous_increase=continous_increase+1
            continous_decrease=0
            continous_zeroes=0
            
            if max_increased_days < continous_increase:
                max_increased_days = continous_increase

        else:
            # if 2 elements are equal & if the value is non-zero->Considered as Continous increase, else as Continous decrease
            if int(current_data[i-1]) == int(current_data[i]):
                if int(current_data[i]) != 0:
                    continous_increase=continous_increase+1
                    continous_decrease=0
                    if max_increased_days < continous_increase:
                        max_increased_days = continous_increase
                    continue
                else:
                    continous_zeroes = continous_zeroes + 1
                    if max_continous_zeroes < continous_zeroes:
                            max_continous_zeroes = continous_zeroes
                        
                
            continous_decrease=continous_decrease+1

            if max_decreased_days < continous_decrease:
                max_decreased_days = continous_decrease

                                  
            #Compute only when declined to zero from a non-zero value     
            if int(current_data[i]) == 0 and int(current_data[i-1]) != 0:
                continous_zeroes=1
                
            continous_increase = 0
            
    statewise_details.update({current_state:[max_increased_days,highest_recorded_cases,max_decreased_days,max_continous_zeroes,last_days_avg]})
print()
print("India Covid-19 Statewise Cases Analysis from ",df[0,1]," to ",df[0,df.shape[1]-1])
print()
for i in statewise_details.items():
    if i[1][0]>=NO_OF_DAYS_TO_VERIFY :
        print(i[0],"::\n","Continous_increased_days-->",i[1][0]," :: highest_recorded_cases_in_a_day-->",i[1][1],"\n",
                           "Continous_decreased_days-->",i[1][2]," :: max_continous_zeroes-->",i[1][3],
                          "\t\t:: last",NO_OF_LAST_DAYS_FOR_AVG,"days avg -->",round(i[1][4],2),"\n",
         )