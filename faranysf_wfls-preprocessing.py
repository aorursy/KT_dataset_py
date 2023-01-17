import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

print('Done')
df = pd.read_csv ("../input/nycwfls/nyc_wfls.csv")
df.rename(columns={"es4": "no_dependents", "es3" : "income", "el17a": "dtl_financial", "el17b": "dtl_jobloss", "el17c": "dtl_employer_dont_offer", 

                   "el17d": "dtl_not_eligible", "el17e": "dtl_readytowork", "el17f": "dtl_nohealthinsurance", "cp1": "familysupport", "cp5": "fsupport_availability", 

                   "el15": "return_samejob", "el16": "unemployment_reason", "ih1": "child_health", "el14": "felt_leavetime", "el13a": "earned_week", 

                   "el13b": "temp_disability_insurnc", "el13c": "maternity_leave", "el13d": "unpaid", "el13e": "other", "mh4": "mother_health", "d4_1":"mother_race", 

                   "de2_5": "spouse_race", "d7": "education"}, inplace=True)
df1 = df[["no_dependents", "income", "dtl_financial", "dtl_jobloss", "dtl_employer_dont_offer", 

                   "dtl_not_eligible", "dtl_readytowork", "dtl_nohealthinsurance", "familysupport", "fsupport_availability", 

                   "return_samejob", "unemployment_reason", "felt_leavetime", "earned_week", 

                   "temp_disability_insurnc", "maternity_leave", "unpaid", 'other', "mother_health", "mother_race", "child_health", 

                   "spouse_race", "education", 'el12mns', 'el12wks', 'cp10wks', 'cp10mns']]

print('Done')
df1.info()
col_catgry = ["no_dependents", "income", "dtl_financial", "dtl_jobloss", "dtl_employer_dont_offer", 

                   "dtl_not_eligible", "dtl_readytowork", "dtl_nohealthinsurance", "familysupport", "fsupport_availability", 

                   "return_samejob", "unemployment_reason", "felt_leavetime", "mother_health", "mother_race", "child_health", 

                   "spouse_race", "education"]

for col in col_catgry: 

    df1[col] = df1[col].astype('category', copy=False)



import warnings

warnings.filterwarnings("ignore")
df1.dtypes
df1.loc[0:5, ['el12mns', 'el12wks']]
df1["el12wks"] = df1["el12wks"]*7

df1["el12mns"] = df1["el12mns"]*30

print('Done')
df1['el12mns'] = df1['el12mns'].replace(np.NaN, 0)

df1['el12wks'] = df1['el12wks'].replace(np.NaN, 0)



df1["leavedays"] = df1.el12wks + df1.el12mns

print("done")
df1.loc[0:5, ['el12mns', 'el12wks', 'leavedays']]
df1 = df1.drop(["el12mns", "el12wks"], axis=1)

print('Done')
df1["cp10wks"] = df1["cp10wks"]*7

df1["cp10mns"] = df1["cp10mns"]*30



df1['cp10wks'] = df1['cp10wks'].replace(np.NaN, 0)

df1['cp10mns'] = df1['cp10mns'].replace(np.NaN, 0)



df1["FmlySLeaveDays"] = df1.cp10mns + df1.cp10wks

print("done")

df1 = df1.drop(["cp10mns", "cp10wks"], axis=1)

print('Done')
df1.fsupport_availability.sample(2) 
return_samejob = df1.return_samejob.value_counts()

print ('Class 1:' , return_samejob[1])

print ('Class 2:' , return_samejob[2])

print ('Class 3:' , return_samejob[3])

print ('Class 4:' , return_samejob[4])

print ('Class 77:' , return_samejob[77])

print('Proportion:', round(return_samejob[1] / return_samejob[2], 2), ':1')
df1.boxplot(column=['leavedays'], by=['income'], figsize = (8,6), patch_artist=True)
df1.isnull().sum()
df1['earned_week'] = df1['earned_week'].replace(np.nan, 0)

df1['temp_disability_insurnc'] = df1['temp_disability_insurnc'].replace(np.nan, 0)

df1['maternity_leave'] = df1['maternity_leave'].replace(np.nan, 0)

df1['unpaid'] = df1['unpaid'].replace(np.nan, 0)

df1['other'] = df1['other'].replace(np.nan, 0)
df1.isnull().sum()