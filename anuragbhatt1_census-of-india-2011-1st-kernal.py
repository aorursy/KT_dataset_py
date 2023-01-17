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
census_data = pd.read_csv('../input/censusindia2011.csv')
census_data.head(5)
df_total=census_data.loc[census_data['TRU']=='Total']
df_total.reset_index(drop=True)
#finding the state having maximum population in TOT_P  not the district part so filter the data again

df_total_states=df_total.loc[df_total['Level']=='STATE']
df_total_states.head(5)
# you can see here that STATE UTTAR PRADESH is having maximum population so now plot these
# and this plot will show which STATE is having lowest population too 
df_total_states = df_total_states.sort_values('TOT_P')

df_total_states.plot.barh('Name','TOT_P',figsize=(10,8),legend=False)

#again we can see that STATE UTTAR PRADESH is having maximum population of males too

df_total_states = df_total_states.sort_values('TOT_M')

df_total_states.plot.barh('Name','TOT_M',figsize=(10,8),legend=False)
#same we can do this for females also so I am not writing it here  
literacy_rate=[]
for x in df_total_states['P_LIT']:
    literacy_rate=df_total_states['P_LIT']/ df_total_states['TOT_P']*100
    
df_total_states['LIT_RATE']=literacy_rate
df_total_states.head()
df_total_states = df_total_states.sort_values('LIT_RATE')

df_total_states.plot.barh('Name','LIT_RATE',figsize=(10,8),legend=False)
# So we can see that KERALA is having highest literacy rate among all the state
#now we will find the litearcy rate among females 
literacy_rate_female=[]
for x in df_total_states['P_LIT']:
    literacy_rate_female=df_total_states['F_LIT']/ df_total_states['P_LIT']*100
    
df_total_states['LIT_RATE_F']=literacy_rate_female
df_total_states.head(5)
df_total_states = df_total_states.sort_values('LIT_RATE_F')

df_total_states.plot.barh('Name','LIT_RATE_F',figsize=(10,8),legend=False)
#Again KERALA is top in female literacy rate  and same way we can find the male literacy rate too
# now finding the illiteracy rate among states
illliteracy_rate=[]
for x in df_total_states['P_ILL']:
    illliteracy_rate=df_total_states['P_ILL']/ df_total_states['TOT_P']*100
    
df_total_states['ILLLIT_RATE']=illliteracy_rate
df_total_states.head(5)
df_total_states = df_total_states.sort_values('ILLLIT_RATE')

df_total_states.plot.barh('Name','ILLLIT_RATE',figsize=(10,8),legend=False)
#Bihar is the state having highest illiterate people living same way we can find it in female male also
#now finding the highest literacy rate among all the district 
#filtering the district from main having only total values not rural and urban subdivision
df_total_district=df_total.loc[df_total['Level']=='DISTRICT']
df_total_district.head(5)
literacy_rate_district=[]
for x in df_total_district['P_LIT']:
    literacy_rate_district=df_total_district['P_LIT']/ df_total_district['TOT_P']*100
    
df_total_district['LIT_RATE']=literacy_rate_district

df_total_district = df_total_district.sort_values('LIT_RATE',ascending=False)


#df_total_district.plot.barh('Name','LIT_RATE',figsize=(10,8),legend=False)
df_temp=df_total_district.head(20)
df_temp.plot.barh('Name','LIT_RATE',figsize=(10,8),legend=False)
# District Pathanamthitta having highest literacy rate
#Finding the district having highest female literacy rate 

literacy_rate_district_female=[]
for x in df_total_district['P_LIT']:
    literacy_rate_district_female=df_total_district['F_LIT']/ df_total_district['P_LIT']*100
    
df_total_district['LIT_RATE_F']=literacy_rate_district_female

df_total_district = df_total_district.sort_values('LIT_RATE_F',ascending=False)


#df_total_district.plot.barh('Name','LIT_RATE',figsize=(10,8),legend=False)
df_temp_f=df_total_district.head(20)
df_temp_f
df_temp_f.plot.barh('Name','LIT_RATE_F',figsize=(10,8),legend=False)
