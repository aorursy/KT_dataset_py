# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as pt
hr = pd.read_csv("../input/core_dataset.csv")
hr.columns = ['Name','EmpNum','State','Zip','Date_of_birth','Age','Sex','MaritalDesc','CitizenDesc','Hispanic/Latino','RaceDesc','Date_of_Hire','Date_of_Termination','Reason_For_Termination','Employment_Status','Department','Position','Pay_Rate','Manager_Name','Employment_Source','Performance_Score']
print(hr)
hr = hr.drop(hr.index[301])
print(hr.columns)
hr_no_dot = hr.drop('Date_of_Termination',axis=1)
PerformanceScore_dict = {'N/A- too early to review': 3,

                        'Needs Improvement': 2,

                        'Fully Meets': 3,

                        '90-day meets': 3,

                        'Exceeds': 4,

                        'Exceptional': 5,

                        'PIP': 1}

hr_no_dot['Performance_Score_Num'] = hr_no_dot['Performance_Score'].replace(PerformanceScore_dict)

print(hr_no_dot)
perf_race = hr_no_dot.set_index(['RaceDesc']).drop(['EmpNum','Zip','Age','Pay_Rate'],axis=1).groupby(['RaceDesc']).mean()

print(perf_race)

perf_race1 = perf_race.reset_index()

print(perf_race1)
x = perf_race1['Performance_Score_Num']

y = perf_race1['RaceDesc']

pt.figure

pt.barh(y,x)
perf_empl_src = hr_no_dot.set_index(['Employment_Source']).drop(['EmpNum','Zip','Age','Pay_Rate'],axis=1).groupby(['Employment_Source']).mean()

print(perf_empl_src)

perf_empl_src1 = perf_empl_src.reset_index()

print(perf_empl_src1)
x = perf_empl_src1['Performance_Score_Num']

y = perf_empl_src1['Employment_Source']

pt.figure

pt.barh(y,x)
perf_dept = hr_no_dot.set_index(['Department']).drop(['EmpNum','Zip','Age','Pay_Rate'],axis=1).groupby(['Department']).mean()

print(perf_dept)

perf_dept1 = perf_dept.reset_index()

print(perf_dept1)
x = perf_dept1['Performance_Score_Num']

y = perf_dept1['Department']

pt.figure

pt.barh(y,x)
pay_race = hr_no_dot.set_index(['RaceDesc']).drop(['EmpNum','Zip','Age','Performance_Score_Num'],axis=1).groupby(['RaceDesc']).mean()

print(pay_race)

pay_race1 = pay_race.reset_index()

print(pay_race1)
x = pay_race1['Pay_Rate']

y = pay_race1['RaceDesc']

pt.figure

pt.barh(y,x)
pay_dept = hr_no_dot.set_index(['Department']).drop(['EmpNum','Zip','Age','Performance_Score_Num'],axis=1).groupby(['Department']).mean()

print(pay_dept)

pay_dept1 = pay_dept.reset_index()

print(pay_dept1)
x = pay_dept1['Pay_Rate']

y = pay_dept1['Department']

pt.figure

pt.barh(y,x)
pay_pos = hr_no_dot.set_index(['Position']).drop(['EmpNum','Zip','Age','Performance_Score_Num'],axis=1).groupby(['Position']).mean()

print(pay_pos)

pay_pos1 = pay_pos.reset_index()

print(pay_pos1)
x = pay_pos1['Pay_Rate']

y = pay_pos1['Position']

pt.figure

pt.barh(y,x)
age_dept = hr_no_dot.set_index(['Department']).drop(['EmpNum','Zip','Pay_Rate','Performance_Score_Num'],axis=1).groupby(['Department']).mean()

print(age_dept)

age_dept1 = age_dept.reset_index()

print(age_dept1)
x = age_dept1['Age']

y = age_dept1['Department']

pt.figure

pt.barh(y,x)